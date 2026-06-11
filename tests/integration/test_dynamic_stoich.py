"""Tests for parameter-dependent stoichiometry.

Exercises the schema, compile, runtime, and AD paths through a small
fixture network whose substrate coefficient is the symbolic expression
``-1/Y``.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin


FIXTURE = Path(__file__).parents[1] / "fixtures" / "dynamic_stoich_network.yaml"


@pytest.fixture
def network():
    return aquakin.load_network_from_file(FIXTURE)


def test_summary_includes_symbolic_coefficient(network):
    """summary() must render parameter-dependent coefficients (evaluated at the
    defaults), not silently drop them by reading the all-zeros static base.

    The fixture's substrate S has the symbolic coefficient ``-1/Y`` (Y=0.5 by
    default, so -2); X is the numeric +1.0. The static base matrix holds 0 for
    S, so a summary reading it would omit S entirely.
    """
    out = network.summary()
    growth_line = next(ln for ln in out.splitlines() if "growth:" in ln)
    assert "S" in growth_line  # would be missing if read from the static base
    assert "2 S" in growth_line  # -1/Y at Y=0.5
    assert "X" in growth_line


def test_dynamic_entries_recorded(network):
    """Compile registers the symbolic stoich entry."""
    assert len(network.stoich_dynamic) == 1
    i, j, _ = network.stoich_dynamic[0]
    assert i == 0  # only one reaction
    assert j == network.species_index["S"]


def test_compute_stoich_at_defaults(network):
    """compute_stoich(params) evaluates ``-1/Y`` at Y=0.5 -> -2.0."""
    stoich = network.compute_stoich(network.default_parameters())
    assert float(stoich[0, network.species_index["S"]]) == pytest.approx(-2.0)
    assert float(stoich[0, network.species_index["X"]]) == 1.0


def test_compute_stoich_responds_to_params(network):
    """Changing Y changes the substrate coefficient."""
    p = network.parameter_values({"Y": 0.25})
    stoich = network.compute_stoich(p)
    # -1/0.25 = -4
    assert float(stoich[0, network.species_index["S"]]) == pytest.approx(-4.0)


def test_dCdt_uses_dynamic_stoich(network):
    """dCdt's substrate term scales with 1/Y."""
    C = jnp.asarray([100.0, 50.0])
    conditions = network.default_conditions()

    # At defaults: rate = mu*S*X = 1*100*50 = 5000; dS/dt = -2 * 5000 = -10000
    dC = network.dCdt(C, network.default_parameters(), conditions.fields, 0)
    assert float(dC[network.species_index["S"]]) == pytest.approx(-10000.0)
    assert float(dC[network.species_index["X"]]) == pytest.approx(5000.0)

    # Halving Y to 0.25 should double the substrate-side magnitude.
    p = network.parameter_values({"Y": 0.25})
    dC2 = network.dCdt(C, p, conditions.fields, 0)
    assert float(dC2[network.species_index["S"]]) == pytest.approx(-20000.0)
    assert float(dC2[network.species_index["X"]]) == pytest.approx(5000.0)


def test_ad_grad_through_stoich_only_param(network):
    """jax.grad through solve w.r.t. Y must be finite and non-zero.

    Y appears only in stoichiometry, so this is the new code path: the
    gradient flows through compute_stoich rather than through rates.
    """
    reactor = aquakin.BatchReactor(network, network.default_conditions())
    C0 = jnp.asarray([100.0, 50.0])

    def loss(params):
        sol = reactor.solve(C0, params, t_span=(0.0, 0.01), t_eval=jnp.asarray([0.0, 0.01]))
        return jnp.sum(sol.C_named("S"))  # final substrate

    g = jax.grad(loss)(network.default_parameters())
    Y_idx = network.param_index["Y"]
    assert jnp.all(jnp.isfinite(g))
    # Increasing Y reduces substrate consumption per unit X formation, so
    # the final S increases -> dS/dY > 0.
    assert float(g[Y_idx]) > 0.0


def test_static_path_unaffected():
    """Networks with purely numeric stoich keep the empty-dynamic path."""
    # ozone_bromate, uv_h2o2, and the SUMO-derived ASM2/3 networks all keep
    # numeric stoich. (ASM1 now uses symbolic stoich for yields/N-content.)
    net = aquakin.load_network("ozone_bromate")
    assert net.stoich_dynamic == []
    # compute_stoich should be a fast path that returns the same matrix.
    assert net.compute_stoich(net.default_parameters()) is net.stoich_matrix


def test_calibration_recovers_yield(network):
    """Generate noisy synthetic obs at a known Y, then fit Y back."""
    reactor = aquakin.BatchReactor(network, network.default_conditions(), rtol=1e-8, atol=1e-10)
    C0 = jnp.asarray([100.0, 50.0])

    # Truth: Y = 0.3, mu = 1.0 (default).
    true_params = network.parameter_values({"Y": 0.3})
    t_obs = jnp.linspace(0.005, 0.05, 10)
    sol = reactor.solve(C0, true_params, t_span=(0.0, 0.05), t_eval=t_obs)
    clean = np.asarray(sol.C_named("X"))
    rng = np.random.default_rng(0)
    sigma = 0.5
    noisy = jnp.asarray(clean + sigma * rng.standard_normal(clean.shape))

    result = aquakin.calibrate(
        reactor,
        C0,
        observations=noisy,
        t_obs=t_obs,
        free_params=["Y"],
        observed_species=["X"],
        loss="nll",
        sigma=jnp.asarray(sigma),
        laplace=False,
    )
    # Y starts at 0.5 (default); should converge near 0.3.
    fit_Y = result.params_named["Y"]
    assert abs(fit_Y - 0.3) < 0.05


def test_invalid_stoich_expression_rejected(tmp_path):
    """A stoich expression referencing a species must be rejected at compile."""
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """\
network: {name: bad, version: "1.0"}
species:
  - {name: A, default_concentration: 1.0}
  - {name: B, default_concentration: 0.0}
parameters:
  k: {value: 1.0, transform: positive_log}
reactions:
  - name: r1
    rate: "k * [A]"
    stoichiometry:
      A: "-1 * [B]"
      B: 1.0
"""
    )
    with pytest.raises(ValueError, match="Stoichiometric coefficient"):
        aquakin.load_network_from_file(bad)


def test_invalid_stoich_function_rejected(tmp_path):
    """Domain functions (monod, etc.) are not allowed in stoich expressions."""
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """\
network: {name: bad, version: "1.0"}
species:
  - {name: A, default_concentration: 1.0}
  - {name: B, default_concentration: 0.0}
parameters:
  k: {value: 1.0, transform: positive_log}
  K: {value: 2.0, transform: positive_log}
reactions:
  - name: r1
    rate: "k * [A]"
    stoichiometry:
      A: "monod(k, K)"
      B: 1.0
"""
    )
    with pytest.raises(ValueError, match="Stoichiometric coefficient"):
        aquakin.load_network_from_file(bad)


def test_unknown_param_in_stoich_rejected(tmp_path):
    """A stoich expression referencing an undeclared parameter is rejected."""
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """\
network: {name: bad, version: "1.0"}
species:
  - {name: A, default_concentration: 1.0}
  - {name: B, default_concentration: 0.0}
parameters:
  k: {value: 1.0, transform: positive_log}
reactions:
  - name: r1
    rate: "k * [A]"
    stoichiometry:
      A: "-1 / nonexistent"
      B: 1.0
"""
    )
    with pytest.raises(KeyError):
        aquakin.load_network_from_file(bad)
