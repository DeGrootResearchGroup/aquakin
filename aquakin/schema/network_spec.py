"""Pydantic models for YAML network files.

These models are used only at load time. Once a ``NetworkSpec`` has been
validated, it is converted to a :class:`~aquakin.core.network.CompiledNetwork`
and Pydantic is not used again on the hot path.
"""

from __future__ import annotations

from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class NetworkMeta(BaseModel):
    """Top-level network metadata (``network:`` block)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str = "1.0"
    description: str = ""
    references: list[str] = Field(default_factory=list)


class SpeciesSpec(BaseModel):
    """One entry of the ``species:`` list."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    units: str = "mol/L"
    default_concentration: float = Field(default=0.0, ge=0.0)


class ConditionSpec(BaseModel):
    """One entry of the ``conditions:`` list."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    default: float = 0.0


_VALID_TRANSFORMS = ("none", "positive_log", "logit")


class PriorSpec(BaseModel):
    """Optional Gaussian prior on a parameter, in physical space.

    Declare the prior either with an explicit ``mean`` + ``std`` (e.g. a
    measured value with reported uncertainty), or with a literature
    ``range: [lo, hi]`` which is converted to a Gaussian centred on the range
    midpoint with ``std = (hi - lo) / 4`` (so the reported range spans about
    ``+-2 sigma``, i.e. ~95% of the prior mass). Used by
    :func:`aquakin.calibrate` to regularise the fit toward literature values,
    which makes otherwise non-identifiable parameter combinations well-posed.
    """

    model_config = ConfigDict(extra="forbid")

    dist: str = "gaussian"
    mean: Optional[float] = None
    std: Optional[float] = None
    range: Optional[tuple[float, float]] = None

    @model_validator(mode="after")
    def _validate(self) -> "PriorSpec":
        if self.dist != "gaussian":
            raise ValueError(f"prior.dist must be 'gaussian', got {self.dist!r}")
        has_mean_std = self.mean is not None and self.std is not None
        has_range = self.range is not None
        if has_mean_std == has_range:
            raise ValueError(
                "prior must declare exactly one of {mean and std} or {range}"
            )
        if has_mean_std and self.std <= 0.0:
            raise ValueError(f"prior std must be > 0, got {self.std}")
        if has_range:
            low, high = self.range
            if not (low < high):
                raise ValueError(f"prior range must satisfy low < high, got {self.range}")
        return self

    def resolved(self) -> tuple[float, float]:
        """Return the Gaussian ``(mean, std)`` in physical space."""
        if self.range is not None:
            low, high = self.range
            return (0.5 * (low + high), (high - low) / 4.0)
        return (float(self.mean), float(self.std))


class ParameterSpec(BaseModel):
    """One entry of a reaction's ``parameters:`` block."""

    model_config = ConfigDict(extra="forbid")

    value: float
    units: str = ""
    bounds: Optional[tuple[float, float]] = None
    transform: str = "none"
    prior: Optional[PriorSpec] = None

    @model_validator(mode="after")
    def _bounds_bracket_value(self) -> "ParameterSpec":
        if self.bounds is not None:
            low, high = self.bounds
            if not (low <= high):
                raise ValueError(f"bounds must satisfy low <= high, got {self.bounds}")
            if not (low <= self.value <= high):
                raise ValueError(
                    f"parameter value {self.value} is outside bounds {self.bounds}"
                )
        return self

    @model_validator(mode="after")
    def _transform_known(self) -> "ParameterSpec":
        if self.transform not in _VALID_TRANSFORMS:
            raise ValueError(
                f"transform must be one of {_VALID_TRANSFORMS}, got {self.transform!r}"
            )
        if self.transform == "positive_log" and self.value <= 0.0:
            raise ValueError(
                f"transform 'positive_log' requires value > 0; got {self.value}"
            )
        if self.transform == "logit" and not (0.0 < self.value < 1.0):
            raise ValueError(
                f"transform 'logit' requires 0 < value < 1; got {self.value}"
            )
        return self


class ReactionSpec(BaseModel):
    """One entry of the ``reactions:`` list."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    reference: str = ""
    rate: str
    parameters: dict[str, ParameterSpec] = Field(default_factory=dict)
    # Each stoichiometric coefficient may be a literal numeric value OR a
    # string expression in the rate-expression grammar that depends only on
    # parameters (no species, conditions, or domain functions). String
    # entries are evaluated at compile / solve time using the actual
    # parameter values, which means yield / N-content / fraction
    # coefficients can be calibrated alongside the kinetic constants.
    stoichiometry: dict[str, Union[float, str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _stoichiometry_non_empty(self) -> "ReactionSpec":
        if not self.stoichiometry:
            raise ValueError(
                f"Reaction '{self.name}' must declare at least one species in "
                f"its stoichiometry; an empty stoichiometry would be a no-op."
            )
        # The "all coefficients are zero" check applies only when every
        # entry is a numeric literal — string expressions may still evaluate
        # non-zero depending on parameter values.
        all_numeric = all(
            isinstance(coef, (int, float)) for coef in self.stoichiometry.values()
        )
        if all_numeric and all(coef == 0 for coef in self.stoichiometry.values()):
            raise ValueError(
                f"Reaction '{self.name}' has all-zero stoichiometric "
                f"coefficients; this reaction would contribute nothing to dC/dt."
            )
        return self


class TotalSpec(BaseModel):
    """One acid/base total in a ``speciation.totals`` entry."""

    model_config = ConfigDict(extra="forbid")

    species: str
    molar_mass: float = Field(gt=0.0)


class StrongAnionSpec(BaseModel):
    """One fully-dissociated strong anion in ``speciation.strong_anions``."""

    model_config = ConfigDict(extra="forbid")

    species: str
    molar_mass: float = Field(gt=0.0)
    charge: float = Field(gt=0.0)


_VALID_TOTAL_KEYS = (
    "carbonate",
    "acetate",
    "propionate",
    "butyrate",
    "valerate",
    "ammonia",
    "phosphate",
    "sulfide",
)


class SpeciationSpec(BaseModel):
    """Optional ``speciation:`` block declaring a state-derived pH field.

    Maps state species onto the acid/base totals consumed by the
    charge-balance pH solver. The produced field (default ``pH``) is computed
    from the instantaneous state on every RHS evaluation and made available to
    ``{pH}`` / ``pH_switch(...)`` rate expressions.
    """

    model_config = ConfigDict(extra="forbid")

    field: str = "pH"
    temperature_field: str = "T"
    temperature_units: str = "celsius"
    z_cation_eq: Union[float, dict[str, str]] = 0.0
    n_iter: int = Field(default=40, ge=1)
    totals: dict[str, TotalSpec] = Field(default_factory=dict)
    strong_anions: list[StrongAnionSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> "SpeciationSpec":
        if self.temperature_units not in ("celsius", "kelvin"):
            raise ValueError(
                f"temperature_units must be 'celsius' or 'kelvin', "
                f"got {self.temperature_units!r}"
            )
        unknown = set(self.totals) - set(_VALID_TOTAL_KEYS)
        if unknown:
            raise ValueError(
                f"speciation.totals has unknown systems {sorted(unknown)}; "
                f"valid keys are {_VALID_TOTAL_KEYS}"
            )
        if isinstance(self.z_cation_eq, dict) and set(self.z_cation_eq) != {"condition"}:
            raise ValueError(
                "speciation.z_cation_eq mapping must have exactly the key "
                "'condition'"
            )
        return self


class PositivityLimiterSpec(BaseModel):
    """Optional ``positivity_limiter:`` block.

    Throttles each species' net reaction term as its concentration approaches
    ``threshold``, preventing negative states and the stiffness they cause.
    """

    model_config = ConfigDict(extra="forbid")

    threshold: float = Field(default=1.0e-3, gt=0.0)


class NetworkSpec(BaseModel):
    """Top-level YAML network file schema."""

    model_config = ConfigDict(extra="forbid")

    network: NetworkMeta
    species: list[SpeciesSpec] = Field(min_length=1)
    conditions: list[ConditionSpec] = Field(default_factory=list)
    parameters: dict[str, ParameterSpec] = Field(default_factory=dict)
    expressions: dict[str, str] = Field(default_factory=dict)
    speciation: Optional[SpeciationSpec] = None
    positivity_limiter: Optional[PositivityLimiterSpec] = None
    reactions: list[ReactionSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_consistency(self) -> "NetworkSpec":
        species_names = [s.name for s in self.species]
        if len(set(species_names)) != len(species_names):
            raise ValueError(f"Duplicate species names: {species_names}")
        condition_names = [c.name for c in self.conditions]
        if len(set(condition_names)) != len(condition_names):
            raise ValueError(f"Duplicate condition names: {condition_names}")
        reaction_names = [r.name for r in self.reactions]
        if len(set(reaction_names)) != len(reaction_names):
            raise ValueError(f"Duplicate reaction names: {reaction_names}")

        species_set = set(species_names)
        for rxn in self.reactions:
            for sp in rxn.stoichiometry:
                if sp not in species_set:
                    raise ValueError(
                        f"Reaction '{rxn.name}' stoichiometry references undeclared "
                        f"species '{sp}'. Declared: {sorted(species_set)}"
                    )

        # Bare-identifier namespace collisions. Within a rate expression the
        # parser sees `name` as a parameter or expression reference; species
        # and conditions are syntactically disambiguated and excluded from
        # this collision set.
        global_params = set(self.parameters.keys())
        expressions = set(self.expressions.keys())
        collide_globals = global_params & expressions
        if collide_globals:
            raise ValueError(
                f"Names appear in both network parameters and expressions: "
                f"{sorted(collide_globals)}"
            )
        for rxn in self.reactions:
            local_params = set(rxn.parameters.keys())
            shadowed = local_params & global_params
            if shadowed:
                raise ValueError(
                    f"Reaction '{rxn.name}' declares local parameter(s) "
                    f"{sorted(shadowed)} that shadow network-level parameter(s). "
                    f"Move the declaration to one place or rename."
                )
            shadowed_expr = local_params & expressions
            if shadowed_expr:
                raise ValueError(
                    f"Reaction '{rxn.name}' declares local parameter(s) "
                    f"{sorted(shadowed_expr)} that collide with named expression(s)."
                )
        return self
