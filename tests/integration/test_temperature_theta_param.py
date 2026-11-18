"""A temperature correction whose ``theta`` is a named model parameter.

First-order decay ``A -> B`` with ``k(T) = k_ref * theta**(T - ref_T)`` has the
closed form ``A(t) = A0 exp(-k(T) t)``, so the theta gradient, the Fisher
information, and the calibration can all be checked against it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin import Experiment
from aquakin.plant.bsm.bsm2 import bsm2_asm1_model
from aquakin.plant.cstr import CSTRUnit
from aquakin.plant.streams import Stream

REF_T = 293.15
K_REF, THETA = 0.3, 1.06
T_OBS = jnp.linspace(0.5, 10.0, 20)

_HEADER = """
model: {name: theta_param, version: "1.0", description: "parameterised theta"}
species:
  - {name: A, units: mol/L, default_concentration: 1.0}
  - {name: B, units: mol/L, default_concentration: 0.0}
  - {name: C, units: mol/L, default_concentration: 1.0}
  - {name: D, units: mol/L, default_concentration: 0.0}
conditions:
  - {name: T, default: 293.15}
"""

_NUMERIC = (
    _HEADER
    + """
parameters:
  k:  {value: 0.3, units: "1/d", temperature: {theta: 1.06, ref_T: 293.15}}
  k2: {value: 0.1, units: "1/d", temperature: {theta: 1.06, ref_T: 293.15}}
reactions:
  - {name: decay, rate: "k * [A]", parameters: {}, stoichiometry: {A: -1, B: 1}}
  - {name: decay2, rate: "k2 * [C]", parameters: {}, stoichiometry: {C: -1, D: 1}}
"""
)

_PARAM = (
    _HEADER
    + """
parameters:
  theta_k: {value: 1.06, units: "-", bounds: [1.0, 1.15]}
  k:  {value: 0.3, units: "1/d", temperature: {theta: theta_k, ref_T: 293.15}}
  k2: {value: 0.1, units: "1/d", temperature: {theta: theta_k, ref_T: 293.15}}
reactions:
  - {name: decay, rate: "k * [A]", parameters: {}, stoichiometry: {A: -1, B: 1}}
  - {name: decay2, rate: "k2 * [C]", parameters: {}, stoichiometry: {C: -1, D: 1}}
"""
)


def _load(text, tmp_path, name):
    p = tmp_path / name
    p.write_text(text)
    return aquakin.load_model_from_file(str(p))


@pytest.fixture
def numeric(tmp_path):
    return _load(_NUMERIC, tmp_path, "numeric.yaml")


@pytest.fixture
def param(tmp_path):
    return _load(_PARAM, tmp_path, "param.yaml")


def _reactor(model, T):
    return aquakin.BatchReactor(model, aquakin.OperatingConditions(T=T))


def _solve(model, T, p=None, t_eval=T_OBS):
    p = model.default_parameters() if p is None else p
    return _reactor(model, T).solve(
        model.default_concentrations(), params=p, t_span=(0.0, 10.0), t_eval=t_eval
    )


# ----- compile-time structure ----------------------------------------------


def test_theta_parameter_is_indexed(param):
    i_k, i_k2, i_t = (param.param_index[n] for n in ("k", "k2", "theta_k"))
    assert param.temperature_theta_params == {i_k: i_t, i_k2: i_t}
    assert "theta_k" in param.parameters
    assert "theta=theta_k" in param.summary()


def test_default_bounds_and_units_for_theta(tmp_path):
    text = _PARAM.replace('theta_k: {value: 1.06, units: "-", bounds: [1.0, 1.15]}',
                          "theta_k: {value: 1.06}")
    net = _load(text, tmp_path, "nobounds.yaml")
    assert net.parameter_bounds["theta_k"] == (1.0, 1.3)
    assert net.parameter_units["theta_k"] == "-"


def test_reaction_local_theta_resolves_in_owning_reaction(tmp_path):
    text = (
        _HEADER
        + """
reactions:
  - name: decay
    rate: "k * [A]"
    parameters:
      th: {value: 1.06}
      k: {value: 0.3, temperature: {theta: th, ref_T: 293.15}}
    stoichiometry: {A: -1, B: 1}
"""
    )
    net = _load(text, tmp_path, "local.yaml")
    assert net.temperature_theta_params == {
        net.param_index["decay.k"]: net.param_index["decay.th"]
    }


# ----- equivalence with the numeric theta ------------------------------------


@pytest.mark.parametrize("T", [REF_T, 283.15, 303.15])
def test_matches_numeric_theta(numeric, param, T):
    a = _solve(numeric, T).C
    b = _solve(param, T).C
    np.testing.assert_allclose(b, a, rtol=1e-12, atol=0.0)


def test_numeric_theta_path_unchanged(numeric):
    """The float path keeps no parameter lookup in its compiled groups."""
    assert numeric.temperature_theta_params == {}
    assert all(g[4] is None and g[5] is None for g in numeric._temp_groups)


# ----- gradients ---------------------------------------------------------------


def test_theta_gradient_matches_finite_differences(param):
    T = 283.15
    i_t = param.param_index["theta_k"]
    i_A = param.species_index["A"]
    p0 = param.default_parameters()
    reactor = _reactor(param, T)
    C0 = param.default_concentrations()

    def final_A(theta):
        p = p0.at[i_t].set(theta)
        return reactor.solve(C0, params=p, t_span=(0.0, 10.0), t_eval=jnp.array([10.0])).C[
            -1, i_A
        ]

    g = float(jax.grad(final_A)(THETA))
    h = 1e-5
    fd = float((final_A(THETA + h) - final_A(THETA - h)) / (2 * h))
    assert np.isfinite(g)
    assert g == pytest.approx(fd, rel=1e-5)
    # Closed form: dA/dtheta = -A * t * k_ref * dT * theta**(dT - 1).
    dT = T - REF_T
    k = K_REF * THETA**dT
    exact = -np.exp(-k * 10.0) * 10.0 * K_REF * dT * THETA ** (dT - 1)
    assert g == pytest.approx(exact, rel=1e-5)


def test_shared_theta_gradient_sums_both_constants(param):
    """theta reaches both rate constants that share it, so its rate gradient is
    the sum of the two per-constant contributions."""
    T = 303.15
    dT = T - REF_T
    C = param.default_concentrations()
    i_t = param.param_index["theta_k"]

    def total_rate(p):
        return jnp.sum(param.rates(C, p, {"T": jnp.array([T])}, 0))

    g = jax.grad(total_rate)(param.default_parameters())[i_t]
    # d/dtheta [(k + k2) * theta**dT] at [A] = [C] = 1.
    exact = (0.3 + 0.1) * dT * THETA ** (dT - 1)
    assert float(g) == pytest.approx(exact, rel=1e-10)


# ----- Fisher information / identifiability ----------------------------------


def _experiment(model, T):
    return Experiment(
        model.default_concentrations(),
        T_OBS,
        sigma=0.01,
        observed_species=["A"],
        reactor=_reactor(model, T),
    )


def test_fisher_reports_theta_and_is_zero_at_reference(param):
    reactor = _reactor(param, REF_T)
    fim = aquakin.fisher_information(
        reactor,
        _experiment(param, REF_T),
        ["k", "theta_k"],
        transforms={"k": "positive_log", "theta_k": "none"},
    )
    assert fim.names == ["k", "theta_k"]
    i = fim.names.index("theta_k")
    assert fim.matrix[i, i] == 0.0
    report = aquakin.identifiability(fim)
    flagged = {
        n for d in report.unidentifiable for n, w in d.loadings.items() if abs(w) > 0.5
    }
    assert "theta_k" in flagged


def test_two_temperatures_make_theta_identifiable(param):
    reactor = _reactor(param, REF_T)
    fim = aquakin.fisher_information(
        reactor,
        [_experiment(param, 283.15), _experiment(param, 298.15)],
        ["k", "theta_k"],
        transforms={"k": "positive_log", "theta_k": "none"},
    )
    report = aquakin.identifiability(fim)
    assert report.unidentifiable == []
    assert np.isfinite(report.std["theta_k"])


# ----- calibration -------------------------------------------------------------


def test_calibrate_recovers_theta(param):
    """One experiment away from ref_T, with k_ref known, identifies theta."""
    T = 283.15
    i_t = param.param_index["theta_k"]
    truth = param.default_parameters().at[i_t].set(1.08)
    obs = _solve(param, T, truth).C_named("A")
    result = aquakin.calibrate(
        _reactor(param, T),
        param.default_concentrations(),
        observations=obs,
        t_obs=T_OBS,
        free_params=["theta_k"],
        observed_species=["A"],
        initial_params=param.default_parameters(),
        laplace=False,
    )
    assert result.params_named["theta_k"] == pytest.approx(1.08, rel=1e-4)


def test_calibrate_recovers_theta_and_k_ref_with_shared_theta(param):
    """k2 is known and shares theta with k, so one experiment off ref_T pins
    theta through C(t) and then k_ref through A(t)."""
    T = 283.15
    truth = (
        param.default_parameters()
        .at[param.param_index["theta_k"]]
        .set(1.08)
        .at[param.param_index["k"]]
        .set(0.45)
    )
    sol = _solve(param, T, truth)
    obs = jnp.stack([sol.C_named("A"), sol.C_named("C")], axis=1)
    result = aquakin.calibrate(
        _reactor(param, T),
        param.default_concentrations(),
        observations=obs,
        t_obs=T_OBS,
        free_params=["k", "theta_k"],
        observed_species=["A", "C"],
        laplace=False,
    )
    assert result.params_named["theta_k"] == pytest.approx(1.08, rel=1e-4)
    assert result.params_named["k"] == pytest.approx(0.45, rel=1e-4)


# ----- validation errors -------------------------------------------------------


def test_missing_theta_parameter(tmp_path):
    text = _PARAM.replace("theta: theta_k, ref_T: 293.15}}\n  k2", "theta: theta_q, ref_T: 293.15}}\n  k2")
    with pytest.raises(ValueError, match="theta_q.*not declared"):
        _load(text, tmp_path, "missing.yaml")


def test_non_positive_theta_parameter(tmp_path):
    text = _PARAM.replace('theta_k: {value: 1.06, units: "-", bounds: [1.0, 1.15]}',
                          "theta_k: {value: 0.0}")
    with pytest.raises(ValueError, match="must be > 0"):
        _load(text, tmp_path, "zero.yaml")


def test_non_positive_theta_lower_bound(tmp_path):
    text = _PARAM.replace("bounds: [1.0, 1.15]", "bounds: [0.0, 1.15]")
    with pytest.raises(ValueError, match="positive lower bound"):
        _load(text, tmp_path, "zerobound.yaml")


def test_temperature_corrected_theta_parameter(tmp_path):
    text = _PARAM.replace(
        'theta_k: {value: 1.06, units: "-", bounds: [1.0, 1.15]}',
        'theta_k: {value: 1.06, temperature: {theta: 1.01, ref_T: 293.15}}',
    )
    with pytest.raises(ValueError, match="cannot be temperature-corrected"):
        _load(text, tmp_path, "corrected.yaml")


# ----- plant / ASM1 -------------------------------------------------------------

_ASM1_THETA = """
model:
  name: asm1_theta_param
  version: "1.0"
  description: "asm1 with a shared autotroph theta parameter"
  extends: asm1
parameters:
  theta_A: {value: 1.107568, units: "-"}
  muA: {temperature: {theta: theta_A}}
  bA: {temperature: {theta: theta_A}}
"""


@pytest.fixture
def asm1_theta(tmp_path):
    return _load(_ASM1_THETA, tmp_path, "asm1_theta.yaml")


def test_cstr_rhs_matches_numeric_theta(asm1_theta):
    asm1 = aquakin.load_model("asm1")
    C = asm1.concentrations(SS=10.0, SO=2.0, SNH=5.0, XB_H=2000.0, XB_A=150.0)

    def rhs(model):
        tank = CSTRUnit(
            name="t", model=model, volume=1000.0, input_port_names=["in"],
            conditions={"T": 285.15},
        )
        s = Stream(Q=jnp.asarray(100.0), C=C, model=model)
        return tank.rhs(jnp.asarray(0.0), C, {"in": s}, model.default_parameters())

    np.testing.assert_allclose(rhs(asm1_theta), rhs(asm1), rtol=1e-12, atol=1e-12)


def test_bsm2_reference_shift_keeps_theta_parameter(asm1_theta):
    """bsm2_asm1_model moves ref_T to 15 degC; the theta parameter survives."""
    shifted = bsm2_asm1_model(asm1_theta)
    assert shifted.temperature_theta_params == asm1_theta.temperature_theta_params
    assert all(ref == pytest.approx(288.15) for _i, _l, ref, _c in shifted.temperature_corrections)
    C = shifted.concentrations(SO=2.0, SNH=5.0, XB_A=150.0)
    i = shifted.reaction_names.index("aerobic_growth_autotrophs")
    p = shifted.default_parameters()
    i_t = shifted.param_index["theta_A"]

    def r(p):
        return shifted.rates(C, p, {"T": jnp.array([278.15])}, 0)[i]

    # theta_A**(-10) scaling at 10 K below the 15 degC reference.
    assert float(r(p)) / float(r(p.at[i_t].set(1.0))) == pytest.approx(1.107568**-10, rel=1e-9)
    assert float(jax.grad(r)(p)[i_t]) < 0.0
