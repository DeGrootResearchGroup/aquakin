"""Plant-level unit audit: is the flowsheet consistent in its declared units?

A plant integrates every unit against **one clock** (``Plant(time_unit=...)``)
and converts what it can before a solve: each kinetic model's rates, each
influent's times, flows and declared concentrations, the time constants built
into the UV reactor, primary clarifier and Takács settler, the ``g/m3`` solids
constants of the thickener and the Takács settler's default parameters, and the
concentrations on a connection between two models that declare the same species
in different units. What is left is checked here. (The ASM1 <-> ADM1 translators
are not audited: they convert their models' declared units themselves, and
``Plant.connect`` rejects a translator built for models whose units differ from
the endpoints'.)

:func:`check_plant_units` (bound as :meth:`Plant.check_units`) audits the units
the plant's parts declare, before any solve:

- **time unit** -- a plant created without ``time_unit`` (deprecated) needs
  its kinetic models to agree on theirs (:attr:`CompiledModel.time_unit`);
- **unit operations** -- a species whose concentration basis a unit's constants
  fix (:class:`~aquakin.plant.units.AssumedUnits`, for a user-written unit that
  cannot convert) must be declared in that basis by the unit's model;
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
from aquakin.utils.units import units_match

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
        The plant's time unit (:attr:`Plant.time_unit`), or ``None`` when
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


def time_unit_errors(plant: Plant) -> list:
    """The findings that make any solve of ``plant`` meaningless: a plant with
    no declared ``time_unit`` whose kinetic models disagree on theirs, so there
    is no clock to convert them to. Used by the solve-time guard."""
    if plant._declared_time_unit is not None:
        return []
    known = {name: u for name, u in _model_time_units(plant).items() if u is not None}
    if len(set(known.values())) <= 1:
        return []
    listing = ", ".join(f"'{n}' in 1/{u}" for n, u in known.items())
    return [
        PlantUnitIssue(
            "error",
            "plant",
            f"the kinetic models disagree on the time unit ({listing}) and the "
            f"plant declares none to convert them to; create it with "
            f"Plant(..., time_unit=...).",
        )
    ]


def _model_time_warnings(model_units: dict) -> list:
    return [
        PlantUnitIssue(
            "warning",
            f"model '{name}'",
            "its time unit cannot be inferred from its parameter units, so its "
            "rates are taken to be per the plant's time unit; declare units: on "
            "its rate constants.",
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
            if species not in model.species_units:
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
    plant_time = plant.time_unit
    issues = time_unit_errors(plant)
    issues += _model_time_warnings(model_units)
    issues += _unit_species_issues(plant)
    if include_models:
        issues += _model_rate_warnings(plant)
    return PlantUnitsReport(time_unit=plant_time, issues=issues)
