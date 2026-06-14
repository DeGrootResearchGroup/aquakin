"""The ``asm1_ammonia_limitation`` variant vs the textbook ``asm1``.

The shipped ``asm1`` is the reference Gujer matrix with no heterotroph ammonia
(nitrogen-source) availability switch. ``asm1_ammonia_limitation`` adds the
BioWin/SUMO-style ``[SNH] / (KNH_H + [SNH])`` factor on both heterotroph growth
rates. These tests pin that the standard model is reference-faithful and that
the variant differs only by that factor.
"""
import jax.numpy as jnp
import pytest

import aquakin


def test_standard_asm1_has_no_ammonia_limitation():
    net = aquakin.load_network("asm1")
    assert "KNH_H" not in net.parameters
    assert net.n_params == 19


def test_variant_adds_ammonia_limitation_only():
    std = aquakin.load_network("asm1")
    lim = aquakin.load_network("asm1_ammonia_limitation")
    # Same state vector and reaction structure -- only KNH_H is added.
    assert lim.species == std.species
    assert lim.reaction_names == std.reaction_names
    assert "KNH_H" in lim.parameters
    assert lim.n_params == std.n_params + 1
    assert set(std.parameters) | {"KNH_H"} == set(lim.parameters)


def _hetero_aerobic_rate(net, snh):
    """Aerobic heterotroph growth rate at a given SNH, everything else fixed."""
    C = net.concentrations(SS=20.0, SO=4.0, SNH=snh, XB_H=1000.0)
    p = net.default_parameters()
    r = net.rates(C, p, {"T": jnp.array([293.15])}, 0)
    return float(r[net.reaction_names.index("aerobic_growth_heterotrophs")])


def test_models_agree_when_ammonia_is_abundant():
    # KNH_H = 0.05; at SNH = 50 the limitation factor is 50/50.05 ~ 1, so the
    # two heterotroph growth rates are practically identical (N-rich domestic
    # influent regime -- the variant is inert there).
    std = aquakin.load_network("asm1")
    lim = aquakin.load_network("asm1_ammonia_limitation")
    assert _hetero_aerobic_rate(lim, 50.0) == pytest.approx(
        _hetero_aerobic_rate(std, 50.0), rel=2e-3)


def test_limitation_bites_when_ammonia_is_low():
    # At low SNH the variant's growth is strictly throttled below the standard
    # model (the factor SNH/(KNH_H+SNH) < 1), and approaches it as SNH grows.
    std = aquakin.load_network("asm1")
    lim = aquakin.load_network("asm1_ammonia_limitation")
    snh = 0.05  # == KNH_H, so the factor is exactly 0.5
    r_std = _hetero_aerobic_rate(std, snh)
    r_lim = _hetero_aerobic_rate(lim, snh)
    assert r_lim == pytest.approx(0.5 * r_std, rel=1e-9)
    assert r_lim < r_std


def test_variant_simulation_runs():
    net = aquakin.load_network("asm1_ammonia_limitation")
    reactor = aquakin.BatchReactor(net, net.default_conditions())
    sol = reactor.solve(
        net.default_concentrations(), params=net.default_parameters(),
        t_span=(0.0, 1.0), t_eval=jnp.linspace(0.0, 1.0, 11),
    )
    assert jnp.all(jnp.isfinite(sol.C))
