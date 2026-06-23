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
    # Optional per-species content of conserved quantities, in the species' own
    # measure -- e.g. ``{COD: 1.0}`` for an organic (1 g COD per g COD),
    # ``{COD: -1.0}`` for dissolved oxygen (an electron acceptor), ``{COD: -2.86,
    # N: 1.0}`` for nitrate-N, ``{COD: 2.0, S: 1.0}`` for sulfide. Quantity names
    # are free-form (``COD`` / ``N`` / ``P`` / ``S`` / ``Fe`` / ``charge`` ...);
    # the conservation check (:meth:`CompiledNetwork.check_conservation`) dots them
    # against the stoichiometry. Advisory metadata: declaring it lets a network
    # carry its own conservation table instead of one hand-maintained elsewhere.
    composition: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _composition_finite(self) -> "SpeciesSpec":
        import math
        for q, v in self.composition.items():
            if not q:
                raise ValueError(
                    f"species '{self.name}' has an empty composition quantity name")
            if not math.isfinite(v):
                raise ValueError(
                    f"species '{self.name}' composition[{q!r}] must be finite; "
                    f"got {v}")
        return self


class ConditionSpec(BaseModel):
    """One entry of the ``conditions:`` list."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    units: str = ""
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


class TemperatureCorrectionSpec(BaseModel):
    """Optional Arrhenius-style temperature correction on a rate constant.

    When present, the parameter is multiplied by ``theta**(T - ref_T)`` during
    rate evaluation, where ``T`` is read from the ``condition`` field. The
    parameter ``value`` is therefore the value *at* ``ref_T`` (the correction is
    unity there). ``ref_T`` is in the same units as the temperature condition
    (Kelvin for the ASM/ADM networks), and a difference is used, so Kelvin and
    Celsius give the same ``theta``.

    ``theta`` is the per-degree factor; for a parameter measured as ``p_hi`` at
    ``T_hi`` and ``p_lo`` at ``T_lo`` it is ``(p_hi / p_lo) ** (1 / (T_hi -
    T_lo))``. The correction is confined to the rate constants — it never
    touches stoichiometric (yield / composition) parameters.
    """

    model_config = ConfigDict(extra="forbid")

    theta: float
    ref_T: float
    condition: str = "T"

    @model_validator(mode="after")
    def _theta_positive(self) -> "TemperatureCorrectionSpec":
        if self.theta <= 0.0:
            raise ValueError(f"temperature.theta must be > 0; got {self.theta}")
        return self


class ParameterSpec(BaseModel):
    """One entry of a reaction's ``parameters:`` block."""

    model_config = ConfigDict(extra="forbid")

    value: float
    units: str = ""
    bounds: Optional[tuple[float, float]] = None
    transform: str = "none"
    prior: Optional[PriorSpec] = None
    temperature: Optional[TemperatureCorrectionSpec] = None

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


class StrongIonSpec(BaseModel):
    """One fully-dissociated strong ion in ``speciation.strong_anions`` or
    ``speciation.strong_cations``. ``charge`` is the magnitude (positive); the
    list it appears in fixes the sign (anionic vs cationic)."""

    model_config = ConfigDict(extra="forbid")

    species: str
    molar_mass: float = Field(gt=0.0)
    charge: float = Field(gt=0.0)


# Single source of truth lives in core/speciation.py (the runtime consumer);
# import it here so the schema validator and the runtime builder can never
# disagree on the valid acid/base total keys.
from aquakin.core.speciation import VALID_TOTAL_KEYS as _VALID_TOTAL_KEYS


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
    activity_model: str = "none"
    # If set, also produce the self-consistent solution ionic strength under this
    # field name, so a precipitation block can share it (see PrecipitationSpec).
    ionic_strength_field: Optional[str] = None
    totals: dict[str, TotalSpec] = Field(default_factory=dict)
    strong_anions: list[StrongIonSpec] = Field(default_factory=list)
    strong_cations: list[StrongIonSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> "SpeciationSpec":
        if self.temperature_units not in ("celsius", "kelvin"):
            raise ValueError(
                f"temperature_units must be 'celsius' or 'kelvin', "
                f"got {self.temperature_units!r}"
            )
        from aquakin.core.ph_solver import _ACTIVITY_MODELS
        if self.activity_model not in _ACTIVITY_MODELS:
            raise ValueError(
                f"speciation.activity_model must be one of {_ACTIVITY_MODELS}; "
                f"got {self.activity_model!r}"
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


# Single source of truth lives in core/precipitation.py (the runtime consumer).
from aquakin.core.precipitation import (
    VALID_PRECIP_FRACTIONS as _VALID_PRECIP_FRACTIONS,
    _PH_SPECIALS as _PRECIP_PH_SPECIALS,
)


class MineralIonSpec(BaseModel):
    """One constituent ion of a mineral in ``precipitation.minerals[].ions``.

    ``count`` is the ion's stoichiometric number in the mineral formula (e.g. 3
    for Ca in Ca3(PO4)2); ``charge`` its charge magnitude (for the activity
    coefficient). ``fraction`` selects how the free activity is obtained: an
    acid/base system (``carbonate``/``phosphate``/``ammonia``/``sulfide``) takes
    the species total times its de/protonated fraction at pH; ``proton`` is H+
    (activity = 10^-pH) and ``hydroxide`` is OH- (activity = Kw/[H+]), neither
    taking a ``species``; omitting it makes the ion a fully-free cation (the
    species total taken as the free ion).
    """

    model_config = ConfigDict(extra="forbid")

    species: Optional[str] = None
    molar_mass: float = Field(default=1.0, gt=0.0)
    count: int = Field(gt=0)
    charge: float = Field(ge=0.0)
    fraction: Optional[str] = None


class MineralSpec(BaseModel):
    """One mineral in a ``precipitation.minerals`` list.

    With the default ``mode: kinetic`` the mineral precipitates / dissolves at
    ``rate_constant * [solid] * sign(sigma) * |sigma|^order`` driven by the
    supersaturation ``sigma`` of the ion-activity product against ``Ksp``,
    exposing the rate factor ``R_<name>``. ``supersaturation_form: bounded``
    swaps the power law for the bounded driver ``tanh(SI/(2 nu) ln10)`` (a ``~k``
    rate Jacobian, for a differentiable dynamic solve of an ultra-insoluble
    mineral). ``pKsp`` is at the reference temperature; ``dH_sp`` is the enthalpy
    of dissolution (J/mol) that van't Hoff-corrects ``Ksp`` with temperature (0,
    the default, leaves ``Ksp`` temperature-independent).

    Declaring ``solid`` (the precipitate species) and ``rate_constant`` makes the
    kinetic precipitation **reaction auto-derived**: the engine consumes each
    constituent ion's ``species`` at ``-count`` and produces ``solid`` at ``+1``,
    with rate ``rate_constant * [solid] * {R_<name>}`` (so the stoichiometry is not
    written a second time). ``solid`` and ``rate_constant`` are set together or
    both omitted (the reaction is then hand-written, referencing ``{R_<name>}``).

    With ``mode: equilibrium`` the mineral is solved to its algebraic saturation
    equilibrium (``IAP = Ksp`` with complementarity, coupled across all
    equilibrium minerals); the engine exposes the equilibrium phase amount
    ``Xeq_<name>`` (in the solid's units) and the reaction is the hand-written
    relaxation toward it. ``solid`` is required (the phase reported), ``order`` /
    ``supersaturation_form`` are unused, and there is no ``rate_constant``."""

    model_config = ConfigDict(extra="forbid")

    name: str
    pKsp: float
    order: float = Field(default=1.0, gt=0.0)
    dH_sp: float = 0.0          # enthalpy of dissolution (J/mol); van't Hoff Ksp(T)
    mode: str = "kinetic"       # "kinetic" (default) or "equilibrium"
    supersaturation_form: str = "power"     # kinetic mode: "power" or "bounded"
    ions: list[MineralIonSpec] = Field(min_length=1)
    solid: Optional[str] = None             # precipitate species
    rate_constant: Optional[ParameterSpec] = None   # crystallisation rate coefficient

    @model_validator(mode="after")
    def _validate_mode(self) -> "MineralSpec":
        if self.mode not in ("kinetic", "equilibrium"):
            raise ValueError(
                f"mineral '{self.name}' mode must be 'kinetic' or 'equilibrium'; "
                f"got {self.mode!r}.")
        if self.supersaturation_form not in ("power", "bounded"):
            raise ValueError(
                f"mineral '{self.name}' supersaturation_form must be 'power' or "
                f"'bounded'; got {self.supersaturation_form!r}.")
        if (self.mode == "kinetic" and self.supersaturation_form == "power"
                and self.order < 1.0):
            # The power driver sign(sigma)*|sigma|^order has derivative
            # order*|sigma|^(order-1) -> infinity as sigma -> 0 (at SI = 0, i.e.
            # equilibrium) when order < 1, so the rate Jacobian is unbounded there
            # and any sensitivity through the equilibrium is non-finite. order >= 1
            # keeps it finite; the 'bounded' (tanh) form has no such restriction.
            raise ValueError(
                f"mineral '{self.name}' has order={self.order} < 1 with the "
                f"'power' supersaturation form, whose rate gradient is infinite at "
                f"saturation (SI=0); use order >= 1, or supersaturation_form: "
                f"'bounded'.")
        if self.mode == "equilibrium":
            if self.solid is None:
                raise ValueError(
                    f"equilibrium-mode mineral '{self.name}' needs a 'solid:' "
                    f"species (the phase its equilibrium amount Xeq_{self.name} is "
                    f"reported for).")
            if self.rate_constant is not None:
                raise ValueError(
                    f"equilibrium-mode mineral '{self.name}' takes no "
                    f"'rate_constant' (its reaction is the relaxation toward "
                    f"Xeq_{self.name}, written by hand).")
        elif (self.solid is None) != (self.rate_constant is None):
            raise ValueError(
                f"mineral '{self.name}': 'solid' and 'rate_constant' must be set "
                f"together (they auto-derive the precipitation reaction), or both "
                f"omitted (the reaction is written by hand).")
        return self


class PrecipitationSpec(BaseModel):
    """Optional ``precipitation:`` block declaring SI-driven mineral precipitation.

    Exposes, per mineral, a saturation index ``SI_<name>`` and a supersaturation
    rate factor ``R_<name>`` as condition fields, computed from the state and the
    system pH (a condition -- e.g. produced by a ``speciation:`` block) on every
    RHS evaluation. A precipitation reaction reads ``{R_<name>}`` in its rate.
    """

    model_config = ConfigDict(extra="forbid")

    pH_field: str = "pH"
    temperature_field: str = "T"
    temperature_units: str = "celsius"
    activity_model: str = "none"
    ionic_strength_offset: float = Field(default=0.0, ge=0.0)
    # If set, read the ionic strength for the activity coefficients from this
    # condition field (e.g. a speciation block's ``ionic_strength_field``)
    # instead of ``ionic_strength_offset`` + the mineral ions -- so the pH and
    # the saturation indices use the same ionic strength.
    ionic_strength_field: Optional[str] = None
    minerals: list[MineralSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate(self) -> "PrecipitationSpec":
        if self.temperature_units not in ("celsius", "kelvin"):
            raise ValueError(
                f"temperature_units must be 'celsius' or 'kelvin', "
                f"got {self.temperature_units!r}")
        from aquakin.core.ph_solver import _ACTIVITY_MODELS
        if self.activity_model not in _ACTIVITY_MODELS:
            raise ValueError(
                f"precipitation.activity_model must be one of {_ACTIVITY_MODELS}; "
                f"got {self.activity_model!r}")
        names = [m.name for m in self.minerals]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate mineral names: {names}")
        # The per-mineral mode / supersaturation_form / solid validation lives in
        # MineralSpec._validate_mode; here we only check the ion declarations.
        for m in self.minerals:
            for ion in m.ions:
                if ion.fraction is not None and ion.fraction not in _VALID_PRECIP_FRACTIONS:
                    raise ValueError(
                        f"mineral '{m.name}' ion fraction {ion.fraction!r} is "
                        f"invalid; valid: {_VALID_PRECIP_FRACTIONS} (or omit).")
                if ion.fraction not in _PRECIP_PH_SPECIALS and ion.species is None:
                    raise ValueError(
                        f"mineral '{m.name}' ion needs a 'species' unless its "
                        f"fraction is one of {_PRECIP_PH_SPECIALS}.")
        return self


def _synthesize_precipitation_reactions(
    precipitation: "PrecipitationSpec", species_set: set
) -> list[ReactionSpec]:
    """Build the precipitation reactions implied by the mineral definitions.

    For each mineral declaring a ``solid`` + ``rate_constant`` (see
    :class:`MineralSpec`), emit a reaction ``<name>_precipitation`` whose rate is
    ``k * [solid] * {R_<name>}`` and whose stoichiometry consumes each ion's
    ``species`` at ``-count`` and produces ``solid`` at ``+1``. Ions with no
    ``species`` (the ``proton`` / ``hydroxide`` specials) carry no mass term.
    Minerals without a ``solid`` are skipped (their reaction is hand-written), as
    are ``mode: equilibrium`` minerals (whose ``solid`` is for the equilibrium
    engine and whose reaction is the hand-written relaxation toward ``Xeq_<name>``).
    """
    out: list[ReactionSpec] = []
    for m in precipitation.minerals:
        if m.solid is None or m.mode == "equilibrium":
            continue
        if m.solid not in species_set:
            raise ValueError(
                f"mineral '{m.name}' solid '{m.solid}' is not a declared species.")
        stoich: dict[str, float] = {}
        for ion in m.ions:
            if ion.species is None:
                continue
            if ion.species not in species_set:
                raise ValueError(
                    f"mineral '{m.name}' ion species '{ion.species}' is not "
                    f"declared.")
            stoich[ion.species] = stoich.get(ion.species, 0.0) - float(ion.count)
        stoich[m.solid] = stoich.get(m.solid, 0.0) + 1.0
        out.append(ReactionSpec(
            name=f"{m.name}_precipitation",
            description=f"Auto-derived SI-driven precipitation / dissolution of {m.name}.",
            rate=f"k * [{m.solid}] * {{R_{m.name}}}",
            parameters={"k": m.rate_constant},
            stoichiometry=stoich,
        ))
    return out


class NetworkSpec(BaseModel):
    """Top-level YAML network file schema."""

    model_config = ConfigDict(extra="forbid")

    network: NetworkMeta
    species: list[SpeciesSpec] = Field(min_length=1)
    conditions: list[ConditionSpec] = Field(default_factory=list)
    parameters: dict[str, ParameterSpec] = Field(default_factory=dict)
    expressions: dict[str, str] = Field(default_factory=dict)
    speciation: Optional[SpeciationSpec] = None
    precipitation: Optional[PrecipitationSpec] = None
    positivity_limiter: Optional[PositivityLimiterSpec] = None
    # Clamp concentrations to >= 0 when evaluating reaction rates (and any
    # state-derived condition). Protects the nonlinear kinetics from a
    # transiently-negative state; the raw state still drives transport and
    # outputs, so it is identity at feasible states. Mirrors the reference
    # IWA/BSM S-function ``xtemp = max(x, 0)`` convention.
    clip_negative_states: bool = False
    # May be empty when every process is an auto-derived precipitation reaction
    # (a mineral with a ``solid`` + ``rate_constant``); _check_consistency
    # synthesizes those and then requires the final list to be non-empty.
    reactions: list[ReactionSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_consistency(self) -> "NetworkSpec":
        species_names = [s.name for s in self.species]
        if len(set(species_names)) != len(species_names):
            raise ValueError(f"Duplicate species names: {species_names}")

        # Auto-derive precipitation reactions from any mineral declaring a
        # ``solid`` + ``rate_constant``, then validate them alongside the
        # hand-written reactions below (species references, name collisions).
        if self.precipitation is not None:
            # Idempotent: this after-validator may run again on a re-validation
            # (e.g. model_validate(model_dump(...)), where the dump already
            # contains the synthesized reactions). Skip any whose name is already
            # present, so the synthesized reactions are not double-appended.
            existing = {r.name for r in self.reactions}
            self.reactions = self.reactions + [
                r for r in _synthesize_precipitation_reactions(
                    self.precipitation, set(species_names))
                if r.name not in existing
            ]
        if not self.reactions:
            raise ValueError(
                "network has no reactions: declare a 'reactions:' list, or a "
                "'precipitation:' block whose minerals carry 'solid' + "
                "'rate_constant' (which auto-derive the reactions).")
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
