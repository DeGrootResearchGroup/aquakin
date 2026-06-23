"""Conservation-derived (`auto` / `?`) stoichiometric coefficients (issue #291,
Phase 2).

A coefficient written ``auto`` is left unknown and solved at compile time from
the reaction's declared conservation laws (its ``conserved_for``, or the network
default), using the per-species ``composition:`` content. These tests pin:

- a single ``auto`` coefficient is solved so the reaction conserves, and the
  resolved value is baked into the (numeric) stoichiometry matrix;
- two coupled ``auto`` coefficients are solved from two balances;
- ``?`` is accepted as an alias for ``auto``;
- the clear errors: no ``conserved_for``, an under-determined system, an
  inconsistent (over-determined) system, and a parameter-expression neighbour
  (deferred to a later phase).
"""

import textwrap

import pytest

import aquakin


def _load(tmp_path, body, name="auto.yaml"):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return aquakin.load_network_from_file(p)


def test_auto_solves_single_coefficient_from_cod(tmp_path):
    # S_S -2, X +1 destroys 1 gCOD, so the O2 electron-acceptor demand must be -1.
    net = _load(tmp_path, """
        network: {name: auto_cod}
        conserved_for: [COD]
        species:
          - {name: S_S, units: gCOD/m3, default_concentration: 1.0, composition: {COD: 1.0}}
          - {name: S_O, units: gO2/m3,  default_concentration: 8.0, composition: {COD: -1.0}}
          - {name: X,   units: gCOD/m3, default_concentration: 1.0, composition: {COD: 1.0}}
        reactions:
          - name: growth
            rate: "mu * [S_S] * [X]"
            parameters: {mu: {value: 1.0}}
            stoichiometry: {S_S: -2.0, S_O: auto, X: 1.0}
        """)
    j = net.species_index["S_O"]
    assert float(net.stoich_matrix[0, j]) == pytest.approx(-1.0, abs=1e-12)
    # The compiled network conserves COD by construction, and the coefficient is
    # numeric (not a parameter-dependent entry).
    assert net.check_conservation(tol=1e-12) == []
    assert net.stoich_dynamic == []


def test_auto_solves_two_coupled_coefficients(tmp_path):
    # Two unknowns (S_O from COD, S_NH from N) solved from the two balances.
    net = _load(tmp_path, """
        network: {name: auto_cod_n}
        species:
          - {name: S_S,  units: gCOD/m3, default_concentration: 1.0, composition: {COD: 1.0, N: 0.05}}
          - {name: S_O,  units: gO2/m3,  default_concentration: 8.0, composition: {COD: -1.0}}
          - {name: S_NH, units: gN/m3,   default_concentration: 1.0, composition: {N: 1.0}}
          - {name: X,    units: gCOD/m3, default_concentration: 1.0, composition: {COD: 1.0, N: 0.10}}
        reactions:
          - name: growth
            conserved_for: [COD, N]
            rate: "mu * [S_S]"
            parameters: {mu: {value: 1.0}}
            stoichiometry: {S_S: -2.0, X: 1.0, S_O: auto, S_NH: auto}
        """)
    assert float(net.stoich_matrix[0, net.species_index["S_O"]]) == pytest.approx(-1.0)
    assert float(net.stoich_matrix[0, net.species_index["S_NH"]]) == pytest.approx(0.0, abs=1e-12)
    assert net.check_conservation(tol=1e-12, quantities=["COD", "N"]) == []


def test_question_mark_is_an_alias(tmp_path):
    net = _load(tmp_path, """
        network: {name: auto_qmark}
        conserved_for: [COD]
        species:
          - {name: S_S, units: gCOD/m3, default_concentration: 1.0, composition: {COD: 1.0}}
          - {name: S_O, units: gO2/m3,  default_concentration: 8.0, composition: {COD: -1.0}}
        reactions:
          - name: oxidation
            rate: "k * [S_S]"
            parameters: {k: {value: 1.0}}
            stoichiometry: {S_S: -1.0, S_O: "?"}
        """)
    assert float(net.stoich_matrix[0, net.species_index["S_O"]]) == pytest.approx(-1.0)


def test_auto_without_conserved_for_raises(tmp_path):
    with pytest.raises(ValueError, match="no quantities to conserve"):
        _load(tmp_path, """
            network: {name: auto_noconserve}
            species:
              - {name: S_S, units: gCOD/m3, default_concentration: 1.0, composition: {COD: 1.0}}
              - {name: S_O, units: gO2/m3,  default_concentration: 8.0, composition: {COD: -1.0}}
            reactions:
              - name: r
                rate: "k * [S_S]"
                parameters: {k: {value: 1.0}}
                stoichiometry: {S_S: -1.0, S_O: auto}
            """)


def test_auto_underdetermined_raises(tmp_path):
    # The auto species (X) carries no content in the only conserved quantity, so
    # the balance cannot constrain it: under-determined.
    with pytest.raises(ValueError, match="under-determined"):
        _load(tmp_path, """
            network: {name: auto_under}
            conserved_for: [COD]
            species:
              - {name: S_S, units: gCOD/m3, default_concentration: 1.0, composition: {COD: 1.0}}
              - {name: X,   units: gN/m3,   default_concentration: 1.0, composition: {N: 1.0}}
            reactions:
              - name: r
                rate: "k * [S_S]"
                parameters: {k: {value: 1.0}}
                stoichiometry: {S_S: -1.0, X: auto}
            """)


def test_auto_inconsistent_overdetermined_raises(tmp_path):
    # One unknown, two balances it cannot satisfy at once: the known side leaves a
    # COD deficit but no N deficit, and the auto species carries both -- so closing
    # COD opens N. Inconsistent.
    with pytest.raises(ValueError, match="cannot conserve all"):
        _load(tmp_path, """
            network: {name: auto_incons}
            conserved_for: [COD, N]
            species:
              - {name: S_S, units: gCOD/m3, default_concentration: 1.0, composition: {COD: 1.0}}
              - {name: X,   units: gCOD/m3, default_concentration: 1.0, composition: {COD: 1.0, N: 1.0}}
            reactions:
              - name: r
                rate: "k * [S_S]"
                parameters: {k: {value: 1.0}}
                stoichiometry: {S_S: -1.0, X: auto}
            """)


def _symbolic_growth(tmp_path):
    # Yield-dependent O2 demand: SS coeff = -1/Y_H, SO solved from COD. The derived
    # coefficient must conserve COD for EVERY Y_H, not just the nominal value.
    return _load(tmp_path, """
        network: {name: auto_paramexpr}
        conserved_for: [COD]
        parameters: {Y_H: {value: 0.67}}
        species:
          - {name: SS,  units: gCOD/m3, default_concentration: 1.0, composition: {COD: 1.0}}
          - {name: SO,  units: gO2/m3,  default_concentration: 8.0, composition: {COD: -1.0}}
          - {name: XBH, units: gCOD/m3, default_concentration: 1.0, composition: {COD: 1.0}}
        reactions:
          - name: growth
            rate: "mu * [SS] * [XBH]"
            parameters: {mu: {value: 1.0}}
            stoichiometry: {SS: "0.0 - 1.0 / Y_H", XBH: 1.0, SO: auto}
        """)


def test_auto_parameter_expression_resolves_and_conserves_for_all_yields(tmp_path):
    import numpy as np
    net = _symbolic_growth(tmp_path)
    # The derived coefficient is parameter-dependent (a stoich_dynamic entry), not
    # a baked constant.
    assert net.stoich_dynamic, "expected a parameter-dependent (dynamic) coefficient"
    j = net.species_index["SO"]
    for Y in (0.4, 0.67, 0.9):
        p = np.array(net.default_parameters())
        p[net.param_index["Y_H"]] = Y
        coef = float(net.compute_stoich(p)[0, j])
        assert coef == pytest.approx((Y - 1.0) / Y, rel=1e-9)   # SO = (Y-1)/Y
        assert net.check_conservation(tol=1e-9, params=p, quantities=["COD"]) == []


def test_auto_parameter_expression_coefficient_is_differentiable(tmp_path):
    """jax.grad flows through the derived (yield-dependent) coefficient: d(SO)/dY_H
    of the resolved -1/Y_H + 1 is +1/Y_H^2."""
    import jax
    import jax.numpy as jnp
    net = _symbolic_growth(tmp_path)
    j, iY = net.species_index["SO"], net.param_index["Y_H"]
    base = net.default_parameters()

    def so_coeff(Y):
        p = base.at[iY].set(Y)
        return net.compute_stoich(p)[0, j]

    g = jax.grad(so_coeff)(0.67)
    assert jnp.isfinite(g)
    assert float(g) == pytest.approx(1.0 / 0.67**2, rel=1e-6)


def test_symbolic_auto_requires_square_system(tmp_path):
    # Two balances (COD, N) but a single auto unknown alongside a parameter
    # expression: the over-determined symbolic consistency would be
    # parameter-dependent, so it is rejected at compile time.
    with pytest.raises(ValueError, match="square system"):
        _load(tmp_path, """
            network: {name: auto_symb_over}
            conserved_for: [COD, N]
            parameters: {Y: {value: 0.5}}
            species:
              - {name: SS, units: gCOD/m3, default_concentration: 1.0, composition: {COD: 1.0, N: 0.05}}
              - {name: SO, units: gO2/m3,  default_concentration: 8.0, composition: {COD: -1.0}}
              - {name: X,  units: gCOD/m3, default_concentration: 1.0, composition: {COD: 1.0, N: 0.10}}
            reactions:
              - name: growth
                rate: "mu * [SS]"
                parameters: {mu: {value: 1.0}}
                stoichiometry: {SS: "0.0 - 1.0 / Y", X: 1.0, SO: auto}
            """)
