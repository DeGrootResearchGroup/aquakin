"""Unit tests for the Pydantic model-file schema validators.

Each test triggers one specific ``raise`` in
:mod:`aquakin.schema.model_spec`, either through the public loader on a
malformed YAML written to ``tmp_path`` or by constructing the Pydantic model
directly.
"""

import textwrap

import pytest

import aquakin
from aquakin.schema.model_spec import (
    MineralIonSpec,
    MineralSpec,
    ModelSpec,
    ParameterSpec,
    PrecipitationSpec,
    PriorSpec,
    ReactionSpec,
    SpeciationSpec,
    SpeciesSpec,
    TemperatureCorrectionSpec,
)


def _write(tmp_path, body: str):
    """Write a dedented YAML body to ``net.yaml`` in ``tmp_path``."""
    p = tmp_path / "net.yaml"
    p.write_text(textwrap.dedent(body))
    return p


# ----- SpeciesSpec.composition (lines 51, 53) ------------------------------


def test_empty_composition_quantity_name_raises():
    """A composition quantity keyed by the empty string is rejected."""
    with pytest.raises(ValueError, match="empty composition"):
        SpeciesSpec(name="A", composition={"": 1.0})


def test_non_finite_composition_value_raises():
    """A non-finite composition content value is rejected."""
    with pytest.raises(ValueError, match="must be finite"):
        SpeciesSpec(name="A", composition={"COD": float("inf")})


# ----- PriorSpec._validate (lines 95, 99, 101, 105) ------------------------


def test_prior_non_gaussian_dist_raises():
    """Only ``dist: gaussian`` is supported (line 95)."""
    with pytest.raises(ValueError, match="dist must be 'gaussian'"):
        PriorSpec(dist="uniform", mean=1.0, std=0.5)


def test_prior_declaring_both_mean_std_and_range_raises():
    """A prior must declare exactly one of {mean,std} or {range} (line 99)."""
    with pytest.raises(ValueError, match="exactly one of"):
        PriorSpec(mean=1.0, std=0.5, range=(0.0, 2.0))


def test_prior_declaring_neither_mean_std_nor_range_raises():
    """Declaring neither also trips the exactly-one check (line 99)."""
    with pytest.raises(ValueError, match="exactly one of"):
        PriorSpec()


def test_prior_non_positive_std_raises():
    """A prior std must be strictly positive (line 101)."""
    with pytest.raises(ValueError, match="prior std must be > 0"):
        PriorSpec(mean=1.0, std=0.0)


def test_prior_range_not_increasing_raises():
    """A prior range must satisfy low < high (line 105)."""
    with pytest.raises(ValueError, match="low < high"):
        PriorSpec(range=(2.0, 1.0))


# ----- ParameterSpec validators (lines 162, 170, 174, 176) -----------------


def test_parameter_bounds_low_above_high_raises():
    """Parameter bounds must satisfy low <= high (line 162)."""
    with pytest.raises(ValueError, match="bounds must satisfy low <= high"):
        ParameterSpec(value=0.5, bounds=(1.0, 0.1))


def test_parameter_unknown_transform_raises():
    """An unknown transform name is rejected (line 170)."""
    with pytest.raises(ValueError, match="transform must be one of"):
        ParameterSpec(value=0.5, transform="sqrt")


def test_parameter_positive_log_requires_positive_value_raises():
    """The positive_log transform requires value > 0 (line 174)."""
    with pytest.raises(ValueError, match="positive_log' requires value > 0"):
        ParameterSpec(value=0.0, transform="positive_log")


def test_parameter_logit_requires_unit_interval_value_raises():
    """The logit transform requires 0 < value < 1 (line 176)."""
    with pytest.raises(ValueError, match="logit' requires 0 < value < 1"):
        ParameterSpec(value=1.5, transform="logit")


# ----- ReactionSpec validators (lines 213, 222) ----------------------------


def test_reaction_empty_stoichiometry_raises():
    """A reaction with an empty stoichiometry is a no-op and rejected (line 213)."""
    with pytest.raises(ValueError, match="at least one species"):
        ReactionSpec(name="r1", rate="k", stoichiometry={})


def test_reaction_all_zero_stoichiometry_raises():
    """All-zero numeric stoichiometric coefficients contribute nothing (line 222)."""
    with pytest.raises(ValueError, match="all-zero stoichiometric"):
        ReactionSpec(name="r1", rate="k", stoichiometry={"A": 0.0, "B": 0.0})


# ----- SpeciationSpec._validate (lines 283, 289, 295, 300) -----------------


def test_speciation_bad_temperature_units_raises():
    """speciation.temperature_units must be celsius or kelvin (line 283)."""
    with pytest.raises(ValueError, match="temperature_units must be"):
        SpeciationSpec(temperature_units="fahrenheit")


def test_speciation_bad_activity_model_raises():
    """speciation.activity_model must be a known model (line 289)."""
    with pytest.raises(ValueError, match="activity_model must be one of"):
        SpeciationSpec(activity_model="bogus")


def test_speciation_unknown_total_system_raises():
    """A totals key outside the valid acid/base systems is rejected (line 295)."""
    with pytest.raises(ValueError, match="unknown systems"):
        SpeciationSpec(totals={"bogus": {"species": "S", "molar_mass": 1.0}})


def test_speciation_z_cation_mapping_bad_key_raises():
    """A z_cation_eq mapping must have exactly the key 'condition' (line 300)."""
    with pytest.raises(ValueError, match="exactly the key 'condition'"):
        SpeciationSpec(z_cation_eq={"foo": "bar"})


# ----- MineralSpec._validate_mode (lines 389, 417) -------------------------


def test_mineral_bad_mode_raises():
    """A mineral mode outside {kinetic, equilibrium} is rejected (line 389)."""
    with pytest.raises(ValueError, match="mode must be 'kinetic' or 'equilibrium'"):
        MineralSpec(
            name="m",
            pKsp=8.0,
            mode="bogus",
            ions=[MineralIonSpec(species="A", count=1, charge=1.0)],
        )


def test_equilibrium_mineral_with_rate_constant_raises():
    """An equilibrium-mode mineral takes no rate_constant (line 417)."""
    with pytest.raises(ValueError, match="takes no"):
        MineralSpec(
            name="m",
            pKsp=8.0,
            mode="equilibrium",
            solid="M_solid",
            rate_constant=ParameterSpec(value=1.0),
            ions=[MineralIonSpec(species="A", count=1, charge=1.0)],
        )


# ----- PrecipitationSpec._validate (lines 457, 463, 469, 480) --------------


def _kinetic_mineral(name="m", solid="M_solid", ions=None):
    """A minimal kinetic mineral with a solid + rate_constant."""
    if ions is None:
        ions = [MineralIonSpec(species="A", count=1, charge=1.0)]
    return MineralSpec(
        name=name,
        pKsp=8.0,
        solid=solid,
        rate_constant=ParameterSpec(value=1.0),
        ions=ions,
    )


def test_precipitation_bad_temperature_units_raises():
    """precipitation.temperature_units must be celsius or kelvin (line 457)."""
    with pytest.raises(ValueError, match="temperature_units must be"):
        PrecipitationSpec(temperature_units="rankine", minerals=[_kinetic_mineral()])


def test_precipitation_bad_activity_model_raises():
    """precipitation.activity_model must be a known model (line 463)."""
    with pytest.raises(ValueError, match="activity_model must be one of"):
        PrecipitationSpec(activity_model="bogus", minerals=[_kinetic_mineral()])


def test_precipitation_duplicate_mineral_names_raises():
    """Duplicate mineral names are rejected (line 469)."""
    with pytest.raises(ValueError, match="duplicate mineral names"):
        PrecipitationSpec(minerals=[_kinetic_mineral(name="dup"), _kinetic_mineral(name="dup")])


def test_precipitation_bad_ion_fraction_raises():
    """An unknown ion fraction is rejected (line 475)."""
    with pytest.raises(ValueError, match="fraction 'bogus' is"):
        PrecipitationSpec(
            minerals=[
                _kinetic_mineral(
                    ions=[MineralIonSpec(species="A", count=1, charge=1.0, fraction="bogus")]
                )
            ]
        )


def test_precipitation_ion_without_species_raises():
    """A non-special ion fraction with no species is rejected (line 480)."""
    with pytest.raises(ValueError, match="needs a 'species'"):
        PrecipitationSpec(
            minerals=[
                _kinetic_mineral(ions=[MineralIonSpec(count=1, charge=1.0, fraction="carbonate")])
            ]
        )


# ----- ModelSpec._check_consistency (lines 577, 584, 587) ------------------


def test_model_no_reactions_raises(tmp_path):
    """A model with no reactions (and no auto-derived precipitation) is rejected
    (line 577)."""
    p = _write(
        tmp_path,
        """
        model: {name: bad}
        species:
          - {name: A, default_concentration: 1.0}
        reactions: []
        """,
    )
    with pytest.raises(ValueError, match="model has no reactions"):
        aquakin.load_model_from_file(p)


def test_model_duplicate_condition_names_raises():
    """Duplicate condition names are rejected (line 584)."""
    with pytest.raises(ValueError, match="Duplicate condition names"):
        ModelSpec(
            model={"name": "bad"},
            species=[{"name": "A", "default_concentration": 1.0}],
            conditions=[{"name": "T"}, {"name": "T"}],
            reactions=[{"name": "r1", "rate": "k", "stoichiometry": {"A": -1}}],
        )


def test_model_duplicate_reaction_names_raises():
    """Duplicate reaction names are rejected (line 587)."""
    with pytest.raises(ValueError, match="Duplicate reaction names"):
        ModelSpec(
            model={"name": "bad"},
            species=[{"name": "A", "default_concentration": 1.0}],
            reactions=[
                {"name": "r1", "rate": "k", "stoichiometry": {"A": -1}},
                {"name": "r1", "rate": "k", "stoichiometry": {"A": -1}},
            ],
        )


# ----- _synthesize_precipitation_reactions (lines 506, 512) ----------------


def test_precipitation_solid_not_declared_species_raises(tmp_path):
    """An auto-derived precipitation reaction whose solid is not a declared
    species is rejected (line 506)."""
    p = _write(
        tmp_path,
        """
        model: {name: bad}
        species:
          - {name: A, default_concentration: 1.0}
        conditions:
          - {name: pH, default: 7.0}
          - {name: T, default: 25.0}
        precipitation:
          minerals:
            - name: m
              pKsp: 8.0
              solid: M_solid
              rate_constant: {value: 1.0}
              ions:
                - {species: A, count: 1, charge: 1.0}
        """,
    )
    with pytest.raises(ValueError, match="is not a declared species"):
        aquakin.load_model_from_file(p)


def test_precipitation_ion_species_not_declared_raises(tmp_path):
    """An auto-derived precipitation reaction whose ion species is not declared
    is rejected (line 512)."""
    p = _write(
        tmp_path,
        """
        model: {name: bad}
        species:
          - {name: M_solid, default_concentration: 0.0}
        conditions:
          - {name: pH, default: 7.0}
          - {name: T, default: 25.0}
        precipitation:
          minerals:
            - name: m
              pKsp: 8.0
              solid: M_solid
              rate_constant: {value: 1.0}
              ions:
                - {species: A, count: 1, charge: 1.0}
        """,
    )
    with pytest.raises(ValueError, match="is not declared"):
        aquakin.load_model_from_file(p)


# ----- TemperatureCorrectionSpec._theta_positive (line 141) ----------------


def test_temperature_correction_non_positive_theta_raises():
    """A temperature correction theta must be strictly positive (line 141)."""
    with pytest.raises(ValueError, match="theta must be > 0"):
        TemperatureCorrectionSpec(theta=0.0, ref_T=293.15)
