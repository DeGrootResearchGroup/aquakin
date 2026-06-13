"""Integration tests for the ASM1 built-in network."""

import jax
import jax.numpy as jnp
import pytest

import aquakin


@pytest.fixture
def network():
    return aquakin.load_network("asm1")


def _build(network, *, t_end=1.0, **C0_overrides):
    conditions = network.default_conditions()
    reactor = aquakin.BatchReactor(network, conditions, rtol=1e-6, atol=1e-9)
    C0 = network.concentrations(C0_overrides)
    sol = reactor.solve(
        C0,
        t_span=(0.0, t_end),
        params=network.default_parameters(),
        t_eval=jnp.linspace(0.0, t_end, 51),
    )
    return sol


def test_shape(network):
    assert network.n_species == 13
    assert network.n_reactions == 8
    assert set(network.species) == {
        "SI", "SS", "XI", "XS", "XB_H", "XB_A", "XP",
        "SO", "SNO", "SNH", "SND", "XND", "SALK",
    }


def test_default_simulation_runs(network):
    """Default initial state should integrate without error over 1 day."""
    sol = _build(network)
    assert jnp.all(jnp.isfinite(sol.C))


def test_aerobic_substrate_consumed(network):
    """Under aerobic conditions with high SS, heterotrophs consume substrate
    and grow during the growth phase (before SS depletes). With default
    ASM1 constants and SS=80, SS depletes within ~0.03 d; once depleted,
    decay dominates and XB_H net decreases. Test on the growth phase."""
    sol = _build(network, t_end=0.01, SO=4.0, SS=80.0, XB_H=600.0)
    ss = sol.C_named("SS")
    xb_h = sol.C_named("XB_H")
    assert float(ss[-1]) < float(ss[0])
    assert float(xb_h[-1]) > float(xb_h[0])


def test_nitrification_oxidises_ammonia(network):
    """Under aerobic conditions with high NH4+ and active autotrophs,
    SNH should decrease and SNO should increase."""
    sol = _build(
        network, t_end=0.5,
        SO=4.0, SNH=30.0, SNO=0.5, XB_A=150.0, XB_H=0.0, SS=0.0,
    )
    snh = sol.C_named("SNH")
    sno = sol.C_named("SNO")
    assert float(snh[-1]) < float(snh[0])
    assert float(sno[-1]) > float(sno[0])


def test_anoxic_denitrification(network):
    """Under anoxic conditions (SO=0) with NO3- and SS available,
    heterotrophs respire nitrate, so SNO decreases."""
    sol = _build(
        network, t_end=0.5,
        SO=0.0, SNO=20.0, SS=60.0, XB_H=600.0, XB_A=0.0,
    )
    sno = sol.C_named("SNO")
    ss = sol.C_named("SS")
    assert float(sno[-1]) < float(sno[0])
    assert float(ss[-1]) < float(ss[0])


def test_decay_only_no_substrate(network):
    """With no SS, no NH4+, no oxygen, biomass undergoes endogenous decay."""
    sol = _build(
        network, t_end=2.0,
        SS=0.0, SO=0.0, SNH=0.0, SNO=0.0, XB_H=500.0, XB_A=100.0,
    )
    assert float(sol.C_named("XB_H")[-1]) < float(sol.C_named("XB_H")[0])
    assert float(sol.C_named("XB_A")[-1]) < float(sol.C_named("XB_A")[0])
    # Decay produces XP and XS.
    assert float(sol.C_named("XP")[-1]) > float(sol.C_named("XP")[0])
    assert float(sol.C_named("XS")[-1]) > float(sol.C_named("XS")[0])


@pytest.mark.slow  # heavy: jax.grad through stiff ASM1 solve
def test_AD_grad_through_solve(network):
    """jax.grad through BatchReactor.solve must produce finite gradients."""
    reactor = aquakin.BatchReactor(network, network.default_conditions())
    C0 = network.default_concentrations()

    def loss(params):
        sol = reactor.solve(
            C0, params=params, t_span=(0.0, 0.2), t_eval=jnp.linspace(0.0, 0.2, 11)
        )
        return jnp.sum(sol.C_named("SNH"))

    g = jax.grad(loss)(network.default_parameters())
    assert jnp.all(jnp.isfinite(g))


def test_transforms_default_to_positive_log(network):
    """Most ASM1 rate constants are declared with positive_log transforms."""
    transforms = network.parameter_transforms
    assert transforms["muH"] == "positive_log"
    assert transforms["etag"] == "logit"
    assert transforms["etah"] == "logit"


@pytest.mark.slow  # heavy: jax.grad to fit a yield
def test_yield_is_calibratable(network):
    """v3 schema: Y_H now appears in the stoichiometry expressions, so
    gradients of any species trajectory w.r.t. Y_H must be non-zero.

    Before this rewrite Y_H was a literature constant frozen at 0.67 in
    the YAML, and calibration could not see it.
    """
    import jax
    reactor = aquakin.BatchReactor(network, network.default_conditions())
    C0 = network.default_concentrations()

    def loss(params):
        sol = reactor.solve(
            C0, params=params, t_span=(0.0, 0.01), t_eval=jnp.linspace(0.0, 0.01, 6)
        )
        return jnp.sum(sol.C_named("SS"))

    g = jax.grad(loss)(network.default_parameters())
    Y_H_idx = network.param_index["Y_H"]
    assert jnp.all(jnp.isfinite(g))
    assert float(g[Y_H_idx]) != 0.0


def test_kinetic_params_are_shared_not_duplicated(network):
    """v3 schema: 15 kinetic constants + 5 stoichiometric (Y_H, Y_A,
    i_XB, i_XP, f_P) = 20 total, each appearing exactly once."""
    assert network.n_params == 20
    # No reaction-namespaced parameter slots — every entry is a bare name.
    assert not any("." in p for p in network.parameters)
    # Stoichiometric parameters live alongside kinetic ones.
    for name in ("Y_H", "Y_A", "i_XB", "i_XP", "f_P"):
        assert name in network.parameters
    assert "muH" in network.parameters
    assert "muH" in network.param_index
