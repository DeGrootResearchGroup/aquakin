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
    with pytest.raises(KeyError, match="influent port 'nope' is not an influent"):
        p._parse_operating([{"kind": "influent_flow", "port": "nope"}])
    with pytest.raises(ValueError, match="operating spec 'kind' must be"):
        p._parse_operating([{"kind": "bogus", "port": "feed"}])


def test_dynamic_operating_is_forward_only():
    """Operating sensitivity rides the augmented variational solve, so reverse mode
    rejects it with a clear message (cheap -- raises before any solve)."""
    from aquakin.plant.bsm import bsm1_warm_start, build_bsm1
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


def test_steady_and_dynamic_operating_sensitivity_agree(cstr_plant):
    """The steady IFT and the dynamic variational stacks compute the SAME kinetic
    and operating sensitivity. On the single CSTR (A -> B at rate k [A], Q/V =
    0.1, feed A = 1) the steady effluent is A* = 0.1 s / (0.1 + k) for a feed
    scale s, so dA*/dk = -0.1 / (0.1 + k)^2 and dA*/ds = 0.1 / (0.1 + k). Started
    at that steady state, the dynamic sensitivity after time T is the steady one
    times 1 - exp(-(0.1 + k) T), which is 1 to round-off at T = 200. Both stacks
    must match the closed form: the steady one to its solve tolerance, the
    dynamic one a decade above its ``rtol``."""
    p = cstr_plant
    base = p.default_parameters()
    k = float(base[p.parameter_index("simple_decay.A_to_B.k")])
    expected = np.array([-0.1 / (0.1 + k) ** 2, 0.1 / (0.1 + k)])
    wrt = ["simple_decay.A_to_B.k"]
    op = [{"kind": "influent_concentration", "port": "feed", "species": "A"}]

    Ss = np.asarray(p.steady_state_sensitivity(
        base, output_fn=lambda y: jnp.array([y[0]]), wrt=wrt, operating=op,
        mode="forward", tol=1e-12))
    np.testing.assert_allclose(Ss[0], expected, rtol=1e-8)

    y_star = p.steady_state(base, tol=1e-12).state
    T = 200.0
    Sd = np.asarray(p.dynamic_sensitivity(
        base, output_fn=lambda sol: jnp.array([sol.C_named("tank", "A")[-1]]),
        t_span=(0.0, T), t_eval=jnp.array([T]), wrt=wrt, operating=op,
        mode="forward", y0=y_star, rtol=1e-8, atol=1e-10))
    np.testing.assert_allclose(Sd[0], expected, rtol=1e-7)


def test_steady_state_sensitivity_reuses_its_compiled_kernel(cstr_plant, monkeypatch):
    """A repeat ``steady_state_sensitivity`` call reuses the compiled IFT kernel
    instead of re-tracing the plant RHS (#920), even with a new ``output_fn``
    object, and still returns the closed form of the single CSTR
    (dA*/dk = -0.1 / (0.1 + k)^2, dA*/ds = 0.1 / (0.1 + k)). Re-tracing calls
    ``plant.derivative``; a cache hit runs the compiled program and calls it
    zero times."""
    p = cstr_plant
    base = p.default_parameters()
    k = float(base[p.parameter_index("simple_decay.A_to_B.k")])
    expected = np.array([-0.1 / (0.1 + k) ** 2, 0.1 / (0.1 + k)])
    wrt = ["simple_decay.A_to_B.k"]
    op = [{"kind": "influent_concentration", "port": "feed", "species": "A"}]
    y_star = p.steady_state(base, tol=1e-12).state

    first = np.asarray(p.steady_state_sensitivity(
        base, state=y_star, output_fn=lambda y: jnp.array([y[0]]), wrt=wrt,
        operating=op, mode="forward"))
    np.testing.assert_allclose(first[0], expected, rtol=1e-8)

    traced = []
    derivative = p.derivative

    def counting_derivative(*args, **kwargs):
        traced.append(1)
        return derivative(*args, **kwargs)

    monkeypatch.setattr(p, "derivative", counting_derivative)
    again = np.asarray(p.steady_state_sensitivity(
        base, state=y_star, output_fn=lambda y: jnp.array([y[0]]), wrt=wrt,
        operating=op, mode="forward"))
    assert traced == []
    np.testing.assert_allclose(again[0], expected, rtol=1e-8)

    # The reverse kernel is a separate compiled program and gives the same
    # closed form, operating column included.
    monkeypatch.undo()
    reverse = np.asarray(p.steady_state_sensitivity(
        base, state=y_star, output_fn=lambda y: jnp.array([y[0]]), wrt=wrt,
        operating=op, mode="reverse"))
    np.testing.assert_allclose(reverse[0], expected, rtol=1e-8)
