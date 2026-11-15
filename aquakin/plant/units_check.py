"""Plant-level unit audit: is the flowsheet consistent in its declared units?

A plant integrates every unit against **one clock** and passes streams between
units **without converting anything**, so a unit mismatch anywhere is silent: a
model whose rate constants are per second next to one per day, an influent file
in ``m3/h``, or a settler whose ``g/m3`` settling parameters are fed ``kg/m3``
solids all produce a plausible-looking but wrong solution. (The ASM1 <-> ADM1
translators are not audited here: they convert their models' declared units
themselves, and ``Plant.connect`` rejects a translator built for models whose
units differ from the endpoints'.)

:func:`check_plant_units` (bound as :meth:`Plant.check_units`) audits the units
the plant's parts declare, before any solve:

- **time unit** -- every kinetic model's rate-constant time unit
  (:attr:`CompiledModel.time_unit`) must agree, and so must the time unit any
  unit operation's internals assume (:class:`~aquakin.plant.units.AssumedUnits`);
- **unit operations** -- a species whose concentration basis a unit's constants
  fix (e.g. a settler's ``g/m3`` settling parameters) must be declared in that
  basis by the unit's model;
- **influents** -- an influent's declared ``units`` (``InfluentSeries.units``)
  must match the plant's time unit, a flow in ``m3`` per that time unit, Kelvin
  temperature, and the model's species units;
- **models** -- each model's own rate-expression check
  (:meth:`CompiledModel.check_units`), reported as warnings.

Every comparison is scale-aware (:func:`aquakin.utils.units.units_match`), so
``mg/L`` matches ``g/m3`` but ``kg/m3`` does not. A unit that is blank or cannot
be parsed is unknown and skipped, so an empty report means "no inconsistency
among the declared units", not a proof. The audit runs in Python on static
metadata only and never touches the compiled solve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aquakin.plant.units import AssumedUnits
from aquakin.utils.units import CELSIUS_SPELLINGS, units_match

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.plant.plant import Plant


@dataclass(frozen=True)
class PlantUnitIssue:
    """One finding of :meth:`Plant.check_units`.

    Attributes
    ----------
    severity : {"error", "warning"}
        ``"error"`` for a mismatch between two *declared* units, which makes the
        solve wrong; ``"warning"`` for an advisory finding (a model's own
        rate-expression check, or a time unit that cannot be determined).
    where : str
        The part of the plant the finding is about, e.g. ``"model 'asm1'"``,
        ``"unit 'uv'"`` or ``"influent 'feed'"``.
    detail : str
        What was declared versus what was expected.
    """

    severity: str
    where: str
    detail: str

    def __str__(self) -> str:
        return f"{self.severity}: [{self.where}] {self.detail}"


@dataclass
class PlantUnitsReport:
    """Result of :meth:`Plant.check_units`.

    Attributes
    ----------
    time_unit : str or None
        The plant's time unit (the one all models agree on), or ``None`` when
        it cannot be determined or the models disagree.
    issues : list of PlantUnitIssue
        Every finding, errors and warnings, in discovery order.
    """

    time_unit: str | None
    issues: list = field(default_factory=list)

    @property
    def errors(self) -> list:
        """The ``"error"`` findings."""
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list:
        """The ``"warning"`` findings."""
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        """True when there is no error (warnings are advisory)."""
        return not self.errors

    def summary(self) -> str:
        """A short human-readable report."""
        lines = [
            f"Plant units: {'OK' if self.ok else 'PROBLEMS'} "
            f"(time unit: {self.time_unit or 'undetermined'})"
        ]
        lines += [f"  {i}" for i in self.errors]
        lines += [f"  {i}" for i in self.warnings]
        return "\n".join(lines)


def _model_time_units(plant: Plant) -> dict:
    return {net.name: net.time_unit for net in plant._ordered_models()}


def _plant_time_unit(model_units: dict) -> str | None:
    known = {u for u in model_units.values() if u is not None}
    return next(iter(known)) if len(known) == 1 else None


def time_unit_errors(plant: Plant) -> list:
    """The findings that make any solve of ``plant`` meaningless: kinetic models
    on different time units, or a unit operation whose internals assume a time
    unit the models do not use. Used by the solve-time guard."""
    model_units = _model_time_units(plant)
    issues = []
    known = {name: u for name, u in model_units.items() if u is not None}
    if len(set(known.values())) > 1:
        listing = ", ".join(f"'{n}' in 1/{u}" for n, u in known.items())
        issues.append(
            PlantUnitIssue(
                "error",
                "plant",
                f"the kinetic models disagree on the time unit ({listing}); every "
                f"model is integrated against the same clock, so the rates of all "
                f"but one are off by the conversion factor.",
            )
        )
        return issues
    plant_time = _plant_time_unit(model_units)
    if plant_time is None:
        return issues
    for name in plant._insertion_order:
        unit = plant.units[name]
        if not isinstance(unit, AssumedUnits):
            continue
        assumed = unit.assumed_units().get("time")
        if assumed is not None and units_match(assumed, plant_time) is False:
            issues.append(
                PlantUnitIssue(
                    "error",
                    f"unit '{name}'",
                    f"its internals assume time in '{assumed}', but the plant's "
                    f"models run in '{plant_time}'.",
                )
            )
    return issues


def _model_time_warnings(model_units: dict) -> list:
    return [
        PlantUnitIssue(
            "warning",
            f"model '{name}'",
            "its time unit cannot be inferred from its parameter units, so the "
            "plant cannot confirm it matches the other models; declare units: "
            "on its rate constants.",
        )
        for name, u in model_units.items()
        if u is None
    ]


def _unit_species_issues(plant: Plant) -> list:
    issues = []
    for name in plant._insertion_order:
        unit = plant.units[name]
        model = getattr(unit, "model", None)
        if not isinstance(unit, AssumedUnits) or model is None:
            continue
        for species, assumed in unit.assumed_units().items():
            if species == "time" or species not in model.species_units:
                continue
            declared = model.species_units[species]
            if units_match(declared, assumed) is False:
                issues.append(
                    PlantUnitIssue(
                        "error",
                        f"unit '{name}'",
                        f"its internals assume species '{species}' in '{assumed}', "
                        f"but model '{model.name}' declares it in '{declared}'.",
                    )
                )
    return issues


def _influent_issues(plant: Plant, plant_time: str | None) -> list:
    issues = []
    for name, series in plant.influents.items():
        declared = series.units or {}
        where = f"influent '{name}'"

        def expect(key, expected, what, declared=declared, where=where):
            unit = declared.get(key)
            if unit is None or expected is None:
                return
            if units_match(unit, expected) is False:
                issues.append(
                    PlantUnitIssue(
                        "error", where, f"declares {what} in '{unit}', expected '{expected}'."
                    )
                )

        expect("t", plant_time, "time 't'")
        expect("Q", None if plant_time is None else f"m3/{plant_time}", "flow 'Q'")
        t_unit = declared.get("T")
        if t_unit is not None and t_unit.strip().lower() in CELSIUS_SPELLINGS:
            issues.append(
                PlantUnitIssue(
                    "error",
                    where,
                    f"declares temperature 'T' in '{t_unit}'; stream temperatures "
                    f"are read in Kelvin, so convert it (add 273.15).",
                )
            )
        else:
            expect("T", "K", "temperature 'T'")
        for species, unit in declared.items():
            if species in ("t", "Q", "T"):
                continue
            model_unit = series.model.species_units.get(species)
            if model_unit is not None and units_match(unit, model_unit) is False:
                issues.append(
                    PlantUnitIssue(
                        "error",
                        where,
                        f"declares species '{species}' in '{unit}', but model "
                        f"'{series.model.name}' declares it in '{model_unit}'.",
                    )
                )
    return issues


def _model_rate_warnings(plant: Plant) -> list:
    issues = []
    for net in plant._ordered_models():
        for w in net.check_units():
            issues.append(PlantUnitIssue("warning", f"model '{net.name}'", str(w)))
    return issues


def check_plant_units(plant: Plant, *, include_models: bool = True) -> PlantUnitsReport:
    """Audit a plant's declared units for consistency (body of
    :meth:`Plant.check_units`).

    Parameters
    ----------
    plant : Plant
        The plant to audit. Nothing is solved or modified.
    include_models : bool, default True
        Also run each kinetic model's own rate-expression check
        (:meth:`CompiledModel.check_units`) and report its findings as warnings.

    Returns
    -------
    PlantUnitsReport
        Errors (declared units that disagree) and warnings (advisory findings).

    Examples
    --------
    >>> report = plant.check_units()
    >>> print(report.summary())
    """
    model_units = _model_time_units(plant)
    plant_time = _plant_time_unit(model_units)
    issues = time_unit_errors(plant)
    issues += _model_time_warnings(model_units)
    issues += _unit_species_issues(plant)
    issues += _influent_issues(plant, plant_time)
    if include_models:
        issues += _model_rate_warnings(plant)
    return PlantUnitsReport(time_unit=plant_time, issues=issues)
