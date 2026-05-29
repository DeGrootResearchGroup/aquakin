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


class ParameterSpec(BaseModel):
    """One entry of a reaction's ``parameters:`` block."""

    model_config = ConfigDict(extra="forbid")

    value: float
    units: str = ""
    bounds: Optional[tuple[float, float]] = None
    transform: str = "none"

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


class NetworkSpec(BaseModel):
    """Top-level YAML network file schema."""

    model_config = ConfigDict(extra="forbid")

    network: NetworkMeta
    species: list[SpeciesSpec] = Field(min_length=1)
    conditions: list[ConditionSpec] = Field(default_factory=list)
    parameters: dict[str, ParameterSpec] = Field(default_factory=dict)
    expressions: dict[str, str] = Field(default_factory=dict)
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
