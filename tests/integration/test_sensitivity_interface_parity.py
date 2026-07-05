"""Parity guard for the steady-state and dynamic sensitivity interfaces.

The two sensitivity stacks are parallel: ``steady_state_sensitivity`` /
``steady_state_dgsm`` (implicit-function-theorem at the operating point) mirror
``dynamic_sensitivity`` / ``dynamic_dgsm`` (augmented variational / discrete
adjoint over a trajectory). They evolved independently, so a capability landed on
one and silently missed the other -- the operating-condition input reached
``solve_sensitivity`` while none of the wrappers exposed it. This test makes such a
divergence loud, the same way ``test_solver_config_single_source.py`` guards the
solver config:

* **Introspective** -- each steady/dynamic twin pair must expose the SAME
  sensitivity-input parameters (``wrt``, ``operating``). If one twin gains an input
  the other lacks, the pair is out of step and this fails.
* **Capability** -- ``operating`` is a unified feature; the two output-sensitivity
  wrappers and the dynamic core must carry it (a regression removing it fails).
* **Behavioural** -- on a constant influent the dynamic operating sensitivity run
  toward steady state equals the steady-state operating sensitivity, so the two
  stacks compute the SAME quantity, not merely accept the same spec.
"""
import inspect

import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.plant.plant import Plant

# Steady <-> dynamic twin method pairs (output sensitivity, and global screen).
_TWINS = [
    ("steady_state_sensitivity", "dynamic_sensitivity"),
    ("steady_state_dgsm", "dynamic_dgsm"),
]
# Sensitivity-input parameters whose presence must match within each twin pair.
_INPUT_PARAMS = ("wrt", "operating")


def _params(method):
    return set(inspect.signature(getattr(Plant, method)).parameters)


@pytest.mark.parametrize("steady, dynamic", _TWINS)
def test_twins_expose_same_sensitivity_inputs(steady, dynamic):
    """Each steady/dynamic twin must expose the SAME sensitivity-input parameters.

    The guard against the two stacks diverging quietly: if one twin gains (or
    loses) ``wrt`` or ``operating`` and the other does not, this fails and points
    at the offending pair.
    """
    ps, pd = _params(steady), _params(dynamic)
    for name in _INPUT_PARAMS:
        on_steady, on_dynamic = name in ps, name in pd
        assert on_steady == on_dynamic, (
            f"{name!r} is exposed by "
            f"{steady if on_steady else dynamic} but not "
            f"{dynamic if on_steady else steady}: the steady and dynamic "
            f"sensitivity interfaces have diverged for this twin pair.")


def test_operating_is_a_unified_capability():
    """``operating`` is the shared operating-condition input; both output-sensitivity
    wrappers and the dynamic core carry it (a regression removing it from any of
    them fails here)."""
    assert "operating" in _params("steady_state_sensitivity")
    assert "operating" in _params("dynamic_sensitivity")
    assert "operating" in _params("solve_sensitivity")


def test_operating_specs_parse_identically():
    """The same operating spec is accepted (and rejected) identically by the shared
    parser every entry point uses, so the stacks cannot drift in what they take."""
    from aquakin.plant.bsm import build_bsm1
    asm1 = aquakin.load_model("asm1")
    p = build_bsm1()
    p.add_influent("feed", asm1.influent({"SNH": 25.0}, Q=18446.0))
    ok = [{"kind": "influent_flow", "port": "feed"},
          {"kind": "influent_concentration", "port": "feed", "species": "SNH"}]
    assert len(p._parse_operating(ok)) == 2
    with pytest.raises(KeyError):
        p._parse_operating([{"kind": "influent_flow", "port": "nope"}])
    with pytest.raises(ValueError):
        p._parse_operating([{"kind": "bogus", "port": "feed"}])


def test_dynamic_operating_is_forward_only():
    """Operating sensitivity rides the augmented variational solve, so reverse mode
    rejects it with a clear message (cheap -- raises before any solve)."""
    from aquakin.plant.bsm import build_bsm1, bsm1_warm_start
    asm1 = aquakin.load_model("asm1")
    p = build_bsm1()
    p.add_influent("feed", asm1.influent({"SNH": 25.0, "SS": 60.0}, Q=18446.0))
    y0 = bsm1_warm_start(p)
    op = [{"kind": "influent_concentration", "port": "feed", "species": "SNH"}]
    with pytest.raises(ValueError, match="forward"):
        p.dynamic_sensitivity(
            p.default_parameters(),
            output_fn=lambda sol: sol.C_named("tank5", "SNH")[-1:],
            t_span=(0.0, 1.0), t_eval=jnp.array([1.0]), wrt=["asm1.muA"],
            operating=op, mode="reverse", y0=y0)


@pytest.mark.slow
def test_steady_and_dynamic_operating_sensitivity_agree():
    """The steady IFT and the dynamic variational stacks compute the SAME operating
    sensitivity: on a constant influent the dynamic effluent-ammonia sensitivity to
    the influent-load scale, run toward steady state, matches the steady-state
    operating column."""
    from aquakin.plant.bsm import build_bsm1, bsm1_warm_start
    asm1 = aquakin.load_model("asm1")
    infl = asm1.influent({"SS": 60., "SNH": 25., "XS": 200., "XB_H": 50.,
                          "SI": 30., "XI": 25., "SND": 6., "XND": 10.,
                          "SALK": 7.}, Q=18446.0)
    p = build_bsm1()
    p.add_influent("feed", infl)
    y0 = bsm1_warm_start(p)
    base = p.default_parameters()
    p._build_state_layout()
    s0, _ = p._state_layout["tank5"]
    snh = s0 + asm1.species_index["SNH"]
    op = [{"kind": "influent_concentration", "port": "feed", "species": "SNH"}]

    Ss = np.asarray(p.steady_state_sensitivity(
        base, output_fn=lambda y: jnp.array([y[snh]]), wrt=["asm1.muA"],
        operating=op, mode="forward"))
    y_star = p.steady_state(base, y0=y0).state
    T = 40.0
    Sd = np.asarray(p.dynamic_sensitivity(
        base, output_fn=lambda sol: jnp.array([sol.C_named("tank5", "SNH")[-1]]),
        t_span=(0.0, T), t_eval=jnp.array([T]), wrt=["asm1.muA"], operating=op,
        mode="forward", y0=y_star, max_steps=200_000))

    # Both the kinetic and the operating column agree at the steady limit.
    assert Sd[0, 0] == pytest.approx(Ss[0, 0], rel=3e-2)   # muA
    assert Sd[0, 1] == pytest.approx(Ss[0, 1], rel=3e-2)   # influent-load scale
