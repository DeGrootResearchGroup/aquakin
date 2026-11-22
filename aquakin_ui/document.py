"""The flowsheet the interface edits: an aquakin spec held as plain data.

:class:`Flowsheet` is the document behind the canvas, with no page or worker in
it, so it can be tested on its own. Units are keyed by their stable identifier
(the plant's unit name); a unit's label is free text that only changes how it is
shown, so nothing that refers to a unit (connections, named streams, a study)
breaks when it is relabelled. :meth:`Flowsheet.to_spec` writes exactly what
``aquakin.Plant.from_spec`` reads.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

from aquakin_ui.catalog import Catalog

SPEC_FORMAT = 2


@dataclass
class Unit:
    """One unit of the flowsheet.

    Parameters
    ----------
    kind : str
        Its catalogue kind.
    settings : dict
        Its settings as plain data (missing ones take the catalogue defaults).
    label : str or None
        Free display text; ``None`` shows the identifier.
    position : dict
        ``{"x", "y"}`` on the canvas.
    """

    kind: str
    settings: dict = field(default_factory=dict)
    label: str | None = None
    position: dict = field(default_factory=lambda: {"x": 0, "y": 0})


def edge_id(c) -> str:
    """The canvas identifier of a connection ``(src, src_port, dst, dst_port)``."""
    return f"{c[0]}.{c[1]}->{c[2]}.{c[3]}"


class Flowsheet:
    """A plant flowsheet: units, connections and the spec's plant-level entries.

    Parameters
    ----------
    catalog : Catalog
        The unit kinds, for ports and defaults.
    name : str, optional
        The plant's name.
    """

    def __init__(self, catalog: Catalog, name: str = "plant"):
        self.catalog = catalog
        self.units: dict[str, Unit] = {}
        self.connections: list[tuple[str, str, str, str]] = []
        # Connection settings the canvas does not edit (a translator, a seed).
        self.extras: dict[str, dict] = {}
        # A new flowsheet runs on a clock of days (aquakin will require one).
        self.meta: dict = {
            "name": name,
            "models": {},
            "streams": {},
            "options": {"time_unit": "d"},
        }
        self.base_dir: Path | None = None

    # -- reading and writing ------------------------------------------------

    @classmethod
    def from_spec(cls, catalog: Catalog, data: dict, base_dir=None) -> Flowsheet:
        """A flowsheet from a spec (``Plant.to_spec`` output or a saved file).

        Raises
        ------
        ValueError
            If the spec is not format 2, a unit's kind is not in the catalogue,
            or a connection names an unknown unit.
        """
        if data.get("format") != SPEC_FORMAT:
            raise ValueError(f"spec format {data.get('format')!r} is not {SPEC_FORMAT}")
        doc = cls(catalog, data.get("name", "plant"))
        doc.base_dir = Path(base_dir) if base_dir is not None else None
        doc.meta = {
            k: v
            for k, v in data.items()
            if k not in ("format", "aquakin", "units", "connections", "layout")
        }
        doc.meta.setdefault("models", {})
        doc.meta.setdefault("streams", {})
        layout = data.get("layout") or {}
        for uid, entry in (data.get("units") or {}).items():
            if entry.get("kind") not in catalog.kinds:
                raise ValueError(
                    f"unit {uid!r} is of kind {entry.get('kind')!r}, which this aquakin "
                    f"does not know"
                )
            x, y = layout.get(uid, (0, 0))
            doc.units[uid] = Unit(
                kind=entry["kind"],
                settings={k: v for k, v in entry.items() if k not in ("kind", "label")},
                label=entry.get("label"),
                position={"x": x, "y": y},
            )
        for entry in data.get("connections") or []:
            if isinstance(entry, str):
                src, dst = (p.strip() for p in entry.split("->"))
                extra = {}
            else:
                src, dst = entry["from"], entry["to"]
                extra = {k: v for k, v in entry.items() if k not in ("from", "to")}
            c = (*doc._endpoint(src, "output"), *doc._endpoint(dst, "input"))
            doc.connections.append(c)
            if extra:
                doc.extras[edge_id(c)] = extra
        if not layout:
            doc.auto_layout()
        return doc

    def to_spec(self, *, layout: bool = True) -> dict:
        """The flowsheet as an aquakin spec (``Plant.from_spec`` reads it as is)."""
        used = {
            m
            for u in self.units.values()
            for m in (u.settings.get("model"), (u.settings.get("reagent") or {}).get("model"))
            if m
        }
        models = {name: ref for name, ref in self.meta.get("models", {}).items() if name in used}
        for name in sorted(used - set(models)):
            models[name] = {"source": "shipped"}
        out = {"format": SPEC_FORMAT, "name": self.meta.get("name", "plant")}
        if self.meta.get("options"):
            out["options"] = self.meta["options"]
        out["models"] = models
        out["units"] = {
            uid: {"kind": u.kind, **({"label": u.label} if u.label else {}), **u.settings}
            for uid, u in self.units.items()
        }
        out["connections"] = [
            {"from": f"{c[0]}.{c[1]}", "to": f"{c[2]}.{c[3]}", **self.extras[edge_id(c)]}
            if self.extras.get(edge_id(c))
            else f"{c[0]}.{c[1]} -> {c[2]}.{c[3]}"
            for c in self.connections
        ]
        out["streams"] = dict(self.meta.get("streams", {}))
        for key in ("influent_endpoint", "effluent_endpoint"):
            if self.meta.get(key):
                out[key] = self.meta[key]
        if layout:
            out["layout"] = {
                uid: [u.position["x"], u.position["y"]] for uid, u in self.units.items()
            }
        return out

    def snapshot(self) -> tuple:
        """Everything an edit can change, copied (for :meth:`restore`, an undo)."""
        return copy.deepcopy((self.units, self.connections, self.extras, self.meta))

    def restore(self, snapshot: tuple) -> None:
        """Go back to what :meth:`snapshot` copied."""
        self.units, self.connections, self.extras, self.meta = copy.deepcopy(snapshot)

    def model_files(self) -> dict[str, str]:
        """``{model name: absolute YAML path}`` for the project's own models."""
        out = {}
        for name, ref in self.meta.get("models", {}).items():
            if ref.get("source", "shipped") != "shipped":
                path = Path(ref["source"])
                if not path.is_absolute() and self.base_dir is not None:
                    path = self.base_dir / path
                out[name] = str(path)
        return out

    def model_names(self) -> list[str]:
        """Every model a unit may choose: the project's own, then the shipped ones."""
        own = [n for n in self.model_files()]
        return own + [n for n in self.catalog.shipped_models if n not in own]

    # -- structure ----------------------------------------------------------

    def ports(self, uid: str) -> tuple[list[str], list[str]]:
        """The ``(inputs, outputs)`` of a unit."""
        u = self.units[uid]
        return self.catalog.ports_for(u.kind, u.settings)

    def _endpoint(self, text: str, side: str) -> tuple[str, str]:
        unit, _, port = text.partition(".")
        if unit not in self.units:
            raise ValueError(f"connection endpoint {text!r} names no unit")
        if not port:
            ins, outs = self.ports(unit)
            ports = outs if side == "output" else ins
            if len(ports) != 1:
                raise ValueError(f"'{unit}' has {len(ports)} {side} ports; name one")
            port = ports[0]
        return unit, port

    def new_id(self, kind: str) -> str:
        """A fresh identifier for a unit of ``kind`` (``cstr1``, ``cstr2``, ...)."""
        i = 1
        while f"{kind}{i}" in self.units:
            i += 1
        return f"{kind}{i}"

    def add(self, kind: str, position: dict, settings: dict | None = None) -> str:
        """Add a unit; returns its new identifier."""
        uid = self.new_id(kind)
        self.units[uid] = Unit(kind=kind, settings=dict(settings or {}), position=dict(position))
        return uid

    def connect(self, src: str, src_port: str, dst: str, dst_port: str) -> str | None:
        """Wire an output to an input; returns why not, or ``None`` when wired."""
        if src == dst:
            return "a unit cannot feed itself"
        if src_port not in self.ports(src)[1] or dst_port not in self.ports(dst)[0]:
            return f"{src}.{src_port} -> {dst}.{dst_port} joins ports that do not exist"
        if any(c[2] == dst and c[3] == dst_port for c in self.connections):
            return f"{dst}.{dst_port} is already fed"
        if any(c[0] == src and c[1] == src_port for c in self.connections):
            return f"{src}.{src_port} already feeds a unit: add a splitter to divide a stream"
        self.connections.append((src, src_port, dst, dst_port))
        return None

    def remove(self, units=(), edges=()) -> None:
        """Remove units (with their connections and references) and connections."""
        units, edges = set(units), set(edges)
        for uid in units:
            self.units.pop(uid, None)
        self.connections = [
            c
            for c in self.connections
            if edge_id(c) not in edges and c[0] in self.units and c[2] in self.units
        ]
        live = {edge_id(c) for c in self.connections}
        self.extras = {k: v for k, v in self.extras.items() if k in live}
        self.meta["streams"] = {
            n: e for n, e in self.meta.get("streams", {}).items() if e.split(".")[0] in self.units
        }
        for key in ("influent_endpoint", "effluent_endpoint"):
            if self.meta.get(key) and self.meta[key].split(".")[0] not in self.units:
                self.meta[key] = None

    def dropped_by(self, uid: str, settings: dict) -> list[str]:
        """The connections giving ``uid`` these settings would remove (its ports change)."""
        ins, outs = self.catalog.ports_for(self.units[uid].kind, settings)
        return [
            f"{c[0]}.{c[1]} -> {c[2]}.{c[3]}"
            for c in self.connections
            if (c[0] == uid and c[1] not in outs) or (c[2] == uid and c[3] not in ins)
        ]

    def apply(self, uid: str, settings: dict, label: str | None) -> list[str]:
        """Replace a unit's settings and label; returns the connections dropped.

        A connection to a port the new settings no longer have (a renamed or
        removed mixer inlet, say) goes with it, and is reported.
        """
        u = self.units[uid]
        u.settings = dict(settings)
        u.label = label or None
        ins, outs = self.ports(uid)
        dropped = [
            c
            for c in self.connections
            if (c[0] == uid and c[1] not in outs) or (c[2] == uid and c[3] not in ins)
        ]
        if dropped:
            self.remove(edges=[edge_id(c) for c in dropped])
        return [f"{c[0]}.{c[1]} -> {c[2]}.{c[3]}" for c in dropped]

    def default_model(self) -> str:
        """The model a new unit starts with: the one most units already use."""
        used = [m for u in self.units.values() if (m := u.settings.get("model"))]
        return max(set(used), key=used.count) if used else "asm1"

    # -- checks and drawing -------------------------------------------------

    def port_status(self) -> tuple[dict, dict]:
        """Per unit, its unfed inputs (errors) and unconnected outputs (products)."""
        fed = {(c[2], c[3]) for c in self.connections}
        used = {(c[0], c[1]) for c in self.connections}
        unfed, dangling = {}, {}
        for uid in self.units:
            ins, outs = self.ports(uid)
            unfed[uid] = [p for p in ins if (uid, p) not in fed]
            dangling[uid] = [p for p in outs if (uid, p) not in used]
        return unfed, dangling

    def back_edges(self, recycles: list[str] | None = None) -> set:
        """The recycle connections: aquakin's (by source endpoint) when known."""
        if recycles is not None:
            sources = set(recycles)
            return {c for c in self.connections if f"{c[0]}.{c[1]}" in sources}
        adj: dict[str, list] = {n: [] for n in self.units}
        for c in self.connections:
            adj[c[0]].append(c)
        state, back = {}, set()

        def visit(n):
            state[n] = 1
            for c in adj[n]:
                if state.get(c[2]) == 1:
                    back.add(c)
                elif c[2] not in state:
                    visit(c[2])
            state[n] = 2

        for n, u in self.units.items():
            if u.kind == "influent" and n not in state:
                visit(n)
        for n in self.units:
            if n not in state:
                visit(n)
        return back

    def lanes(self, back: set, node_height: float = 120.0) -> dict:
        """A lane below the units each recycle spans, stacked so lanes do not overlap."""
        lanes: dict = {}

        def span(c):
            return abs(self.units[c[0]].position["x"] - self.units[c[2]].position["x"])

        for c in sorted(back, key=span):
            src, dst = self.units[c[0]].position, self.units[c[2]].position
            x0, x1 = sorted((src["x"], dst["x"]))
            low = max(src["y"], dst["y"])
            # Below both ends, and below every unit in between down to the lower end
            # (on a wrapped layout the two ends can sit in different bands).
            ys = [low] + [
                u.position["y"]
                for u in self.units.values()
                if x0 - 10 <= u.position["x"] <= x1 + 10 and u.position["y"] <= low
            ]
            floor = max(ys) + node_height + 20
            lanes[c] = max([floor] + [v + 26 for v in lanes.values()])
        return lanes

    def auto_layout(self, per_band: int = 7) -> None:
        """Arrange the units left to right along the forward connections.

        Units are ranked by their longest forward path from a source, and a
        long line is wrapped into bands of ``per_band`` ranks so a plant like
        BSM2 stays readable when the whole of it is in view.
        """
        back = self.back_edges()
        rank = dict.fromkeys(self.units, 0)
        for _ in self.units:
            for c in self.connections:
                if c not in back:
                    rank[c[2]] = max(rank[c[2]], rank[c[0]] + 1)
        rows: dict[int, int] = {}
        for uid in self.units:
            rows[rank[uid]] = rows.get(rank[uid], 0) + 1
        band_height: dict[int, int] = {}
        for r, n in rows.items():
            band_height[r // per_band] = max(band_height.get(r // per_band, 0), n)
        band_y, y = {}, 0
        for band in sorted(band_height):
            band_y[band] = y
            y += band_height[band] * 170 + 230
        placed: dict[int, int] = {}
        for uid in self.units:
            r = rank[uid]
            row = placed.get(r, 0)
            placed[r] = row + 1
            self.units[uid].position = {
                "x": (r % per_band) * 240,
                "y": band_y[r // per_band] + row * 170,
            }
