"""Assign real unit strings to the SUMO-derived ASM networks (issue #199).

The four SUMO-import networks (``asm2d``, ``asm2d_tud``, ``asm3``,
``asm3_biop``) shipped with placeholder parameter units (``"0"`` /
``"SmallNumber"`` / ``"-BigNumber"`` -- SUMO's internal sentinels, copied
verbatim by ``scripts/sumo_to_aquakin.py``) and species units in an unparseable
dotted dialect (``g COD.m-3``, ``- g COD.m-3`` for oxygen, ``kmol HCO3-.m-3``).
That left ``network.time_unit`` ``None`` and ``check_units`` unable to check a
single reaction.

This post-processor rewrites only the ``units: "..."`` field on each species and
parameter line (a surgical per-line text substitution -- everything else stays
byte-identical, so the diff is exactly the unit strings). It is the
units-stamping companion to ``scripts/sumo_to_aquakin.py`` (which the source
SUMO JSON dumps -- not shipped -- are fed through), analogous to the
``_make_khalil_*.py`` post-processors. Re-run after a regeneration:

    cd aquakin/networks && python _fix_sumo_units.py

Units are assigned from each network's *own structure*, so the result is
verifiable: ``check_units`` is clean afterwards except for the handful of
reactions whose SUMO rate form is irreducibly cross-currency (the ASM2d-TUD
biomass-normalised PP-storage rates ``qPP * XPAO**2 / XPP * ...``), which the
root check correctly flags and which are documented, not fixed here.

  * species   -- the dotted dialect is normalised to the parseable
                 ``g_<currency>/m3`` form; oxygen (SUMO's negative-COD ``- g
                 COD.m-3`` or ``g O2.m-3``) becomes ``g_O2/m3`` to match the
                 hand-written ``asm1`` convention; ``kmol HCO3-.m-3`` -> ``kmol/m3``.
  * half-saturation constants -- the currency of the species they limit, read
                 from the Monod term that uses them (so they always match).
  * rate constants -- ``1/d``; the one second-order constant ``kPRE``
                 (precipitation, ``kPRE * [SPO4] * [XMeOH]``) is ``m3/(g_TSS*d)``.
  * correction factors (``eta*``) and saturation *ratios* (``KX``) -- ``-``.
  * yields / composition / charge constants -- best-effort currency ratios;
                 these appear only in stoichiometry, never in a rate, so they
                 are advisory (``check_units`` never reads them).
"""
from __future__ import annotations

import re
from pathlib import Path

import aquakin
from aquakin.core.nodes import (
    MonodInhibitionNode,
    MonodInhibitionRatioNode,
    MonodNode,
    MonodRatioNode,
    ParamNode,
    SpeciesNode,
)

NETWORKS = ["asm2d", "asm2d_tud", "asm3", "asm3_biop"]

# Species-unit currency token, for building half-saturation / ratio units.
_CURRENCY_OF = {
    "g_COD/m3": "COD", "g_N/m3": "N", "g_P/m3": "P",
    "g_O2/m3": "O2", "g_TSS/m3": "TSS", "kmol/m3": None,
}


def normalise_species_unit(raw: str) -> str:
    """SUMO dotted dialect -> parseable ``g_<currency>/m3``."""
    s = raw.replace("⁻", "-").replace("³", "3").replace("·", ".").strip()
    if "O2" in s or s.startswith("-"):
        return "g_O2/m3"            # dissolved oxygen (SUMO writes it as -COD)
    if "TSS" in s:
        return "g_TSS/m3"
    if "HCO3" in s or "kmol" in s:
        return "kmol/m3"            # alkalinity
    if s.startswith("g N"):
        return "g_N/m3"
    if s.startswith("g P"):
        return "g_P/m3"
    if "g COD" in s:
        return "g_COD/m3"
    return raw


def _walk(node):
    yield node
    for c in node.children():
        yield from _walk(c)


def _pname(n):
    return n.name.split(".", 1)[-1] if isinstance(n, ParamNode) else None


def _sname(n):
    return n.name if isinstance(n, SpeciesNode) else None


def _ratio_unit(unit_a: str, unit_b: str) -> str:
    """Unit of ``A/B`` for two ``g_<cur>/m3`` species units (the ``/m3`` cancels)."""
    ca, cb = _CURRENCY_OF.get(unit_a), _CURRENCY_OF.get(unit_b)
    if ca is not None and ca == cb:
        return "-"
    if ca is not None and cb is not None:
        return f"g_{ca}/g_{cb}"
    return "-"


def _advisory_unit(name: str) -> str:
    """Best-effort unit for a stoichiometry-only constant (never rate-checked)."""
    if name.startswith("f"):
        return "-"
    if name.startswith("iN_"):
        return "g_N/g_COD"
    if name.startswith("iP_"):
        return "g_P/g_COD"
    if name.startswith("iTSS_") or name.startswith("iSS_"):
        return "g_TSS/g_COD"
    if name == "iCOD_NO3":
        return "g_COD/g_N"
    if name == "iNO3_N2":
        return "-"
    if name.startswith("iCharge_"):
        if name.endswith("SPO4") or name.endswith("XPAO_PP"):
            return "mol/g_P"
        if name.endswith("SVFA"):
            return "mol/g_COD"
        return "mol/g_N"
    if name in ("YA", "YAUT"):
        return "g_COD/g_N"
    if name.startswith("YPO") or name.startswith("YPP"):
        return "g_P/g_COD"
    if name.startswith("Y"):
        return "g_COD/g_COD"
    return "-"


def assign_units(network):
    """Return ``(species_units, parameter_units)`` for a compiled SUMO network."""
    species_units = {s: normalise_species_unit(network.units_of(s))
                     for s in network.species}
    # half-saturation constants: currency of the species they limit
    kmonod, in_rate = {}, set()
    for ast in network.rate_asts:
        for n in _walk(ast):
            if isinstance(n, ParamNode):
                in_rate.add(_pname(n))
            if isinstance(n, (MonodNode, MonodInhibitionNode)):
                k, s = _pname(n.K), _sname(n.X)
                if k and s in species_units:
                    kmonod[k] = species_units[s]
            elif isinstance(n, (MonodRatioNode, MonodInhibitionRatioNode)):
                k, a, b = _pname(n.K), _sname(n.A), _sname(n.B)
                if k and a in species_units and b in species_units:
                    kmonod[k] = _ratio_unit(species_units[a], species_units[b])
    parameter_units = {}
    for p in network.parameters:
        if p in kmonod:
            parameter_units[p] = kmonod[p]
        elif p.startswith("eta") or p.startswith("KX"):
            parameter_units[p] = "-"          # correction factor / ratio saturation
        elif p == "kPRE":
            parameter_units[p] = "m3/(g_TSS*d)"   # second-order precipitation
        elif p in in_rate:
            parameter_units[p] = "1/d"        # rate constant
        else:
            parameter_units[p] = _advisory_unit(p)
    return species_units, parameter_units


# A species line: ``- {name: SO2, ..., units: "...", ...}``.
_SPECIES_LINE = re.compile(r"\bname:\s*([A-Za-z_]\w*)\b")
# A parameter line: ``  KO2_H: {value: ..., units: "..."}``.
_PARAM_LINE = re.compile(r"^\s*([A-Za-z_]\w*):\s*\{")
_UNITS_FIELD = re.compile(r'units:\s*"[^"]*"')


def _rewrite_units(line: str, species_units, parameter_units) -> str:
    """Replace only the ``units: "..."`` field on one YAML line, by name."""
    if "units:" not in line:
        return line
    m = _PARAM_LINE.match(line)
    if m and m.group(1) in parameter_units:
        new = parameter_units[m.group(1)]
    else:
        m = _SPECIES_LINE.search(line)
        if m and m.group(1) in species_units:
            new = species_units[m.group(1)]
        else:
            return line
    return _UNITS_FIELD.sub(f'units: "{new}"', line, count=1)


def main():
    here = Path(__file__).parent
    for name in NETWORKS:
        net = aquakin.load_network(name)
        species_units, parameter_units = assign_units(net)
        path = here / f"{name}.yaml"
        lines = path.read_text().splitlines(keepends=True)
        out = [_rewrite_units(ln, species_units, parameter_units) for ln in lines]
        path.write_text("".join(out))
        print(f"  {name}: stamped {len(species_units)} species + "
              f"{len(parameter_units)} parameter units")


if __name__ == "__main__":
    main()
