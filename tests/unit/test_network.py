"""CompiledNetwork unit tests."""

import jax.numpy as jnp
import pytest

import aquakin


def test_stoich_matrix_shape(simple_network):
    assert simple_network.stoich_matrix.shape == (1, 2)
    # A -> B with stoich A:-1, B:+1
    a_idx = simple_network.species_index["A"]
    b_idx = simple_network.species_index["B"]
    assert float(simple_network.stoich_matrix[0, a_idx]) == -1.0
    assert float(simple_network.stoich_matrix[0, b_idx]) == 1.0


def test_default_arrays(simple_network):
    C0 = simple_network.default_concentrations()
    assert C0.shape == (2,)
    p0 = simple_network.default_parameters()
    assert p0.shape == (1,)
    assert float(p0[0]) == pytest.approx(0.1)


def test_namespaced_param_keys(simple_network):
    assert "A_to_B.k" in simple_network.param_index
    assert simple_network.parameters == ["A_to_B.k"]


def test_rates_signature(simple_network):
    C = jnp.asarray([1.0, 0.0])
    params = simple_network.default_parameters()
    conditions = aquakin.SpatialConditions.uniform(1, T=293.15)
    r = simple_network.rates(C, params, conditions.fields, 0)
    assert r.shape == (1,)
    assert float(r[0]) == pytest.approx(0.1)


def test_dCdt(simple_network):
    C = jnp.asarray([1.0, 0.0])
    params = simple_network.default_parameters()
    conditions = aquakin.SpatialConditions.uniform(1, T=293.15)
    rhs = simple_network.dCdt(C, params, conditions.fields, 0)
    # dA/dt = -0.1, dB/dt = +0.1
    assert float(rhs[0]) == pytest.approx(-0.1)
    assert float(rhs[1]) == pytest.approx(0.1)


def test_summary_smoke(simple_network):
    out = simple_network.summary()
    assert "simple_decay" in out
    assert "A_to_B" in out


def test_species_units_and_descriptions(simple_network):
    """Species units/descriptions are carried from the YAML through compile."""
    assert simple_network.species_units == {"A": "mol/L", "B": "mol/L"}
    assert simple_network.species_descriptions == {"A": "Reactant", "B": "Product"}

    assert simple_network.units_of("A") == "mol/L"
    assert simple_network.description_of("B") == "Product"

    with pytest.raises(KeyError):
        simple_network.units_of("nope")
    with pytest.raises(KeyError):
        simple_network.description_of("nope")


def test_summary_includes_units(simple_network):
    out = simple_network.summary()
    # Each species line carries its units in brackets and its description.
    assert "[mol/L]" in out
    assert "Reactant" in out


def test_time_unit_seconds(simple_network):
    """The fixture's rate constant is ``s-1``, so the network is in seconds."""
    assert simple_network.time_unit == "s"


def test_summary_includes_time_unit(simple_network):
    out = simple_network.summary()
    assert "Time unit: s" in out
    assert "seconds" in out


_DAYS_YAML = """
network:
  name: days_net
  description: "Rate constant in 1/d."
species:
  - {name: A, units: g/m3, default_concentration: 1.0}
  - {name: B, units: g/m3, default_concentration: 0.0}
conditions:
  - {name: T, default: 293.15}
reactions:
  - name: A_to_B
    rate: "k * [A]"
    parameters:
      k: {value: 0.1, units: "1/d"}
    stoichiometry: {A: -1, B: +1}
"""

# A second-order rate (time at -1) plus a half-saturation constant with no time:
# the inverse-time token is unambiguous (``d``).
_DAYS_SECOND_ORDER_YAML = """
network:
  name: days_net2
  description: "Second-order rate plus a non-time parameter."
species:
  - {name: A, units: g/m3, default_concentration: 1.0}
  - {name: B, units: g/m3, default_concentration: 0.0}
conditions:
  - {name: T, default: 293.15}
reactions:
  - name: A_to_B
    rate: "k * [A] * monod([A], Khalf)"
    parameters:
      k: {value: 0.1, units: "m3/(g*d)"}
      Khalf: {value: 1.0, units: "g/m3"}
    stoichiometry: {A: -1, B: +1}
"""

# Mixed inverse-time tokens (one rate in 1/d, one in 1/s) -> ambiguous.
_AMBIGUOUS_YAML = """
network:
  name: mixed_net
  description: "Rate constants disagree on the time unit."
species:
  - {name: A, units: g/m3, default_concentration: 1.0}
  - {name: B, units: g/m3, default_concentration: 0.0}
conditions:
  - {name: T, default: 293.15}
reactions:
  - name: A_to_B
    rate: "k1 * [A]"
    parameters:
      k1: {value: 0.1, units: "1/d"}
    stoichiometry: {A: -1, B: +1}
  - name: B_to_A
    rate: "k2 * [B]"
    parameters:
      k2: {value: 0.1, units: "1/s"}
    stoichiometry: {B: -1, A: +1}
"""

# No declared time units at all -> cannot infer.
_NO_UNITS_YAML = """
network:
  name: bare_net
  description: "No parameter units declared."
species:
  - {name: A, units: g/m3, default_concentration: 1.0}
  - {name: B, units: g/m3, default_concentration: 0.0}
conditions:
  - {name: T, default: 293.15}
reactions:
  - name: A_to_B
    rate: "k * [A]"
    parameters:
      k: {value: 0.1}
    stoichiometry: {A: -1, B: +1}
"""


def _net_from_yaml(text, tmp_path):
    path = tmp_path / "net.yaml"
    path.write_text(text)
    return aquakin.load_network_from_file(path)


def test_time_unit_days(tmp_path):
    assert _net_from_yaml(_DAYS_YAML, tmp_path).time_unit == "d"


def test_time_unit_days_from_second_order_rate(tmp_path):
    """A non-time parameter (half-saturation constant) does not confuse it; the
    one inverse-time token from the second-order rate constant is used."""
    assert _net_from_yaml(_DAYS_SECOND_ORDER_YAML, tmp_path).time_unit == "d"


def test_time_unit_ambiguous_returns_none(tmp_path):
    """Rate constants that disagree on the time unit give ``None``, not a guess."""
    assert _net_from_yaml(_AMBIGUOUS_YAML, tmp_path).time_unit is None


def test_time_unit_none_when_undeclared(tmp_path):
    assert _net_from_yaml(_NO_UNITS_YAML, tmp_path).time_unit is None


def test_summary_time_unit_unknown(tmp_path):
    out = _net_from_yaml(_NO_UNITS_YAML, tmp_path).summary()
    assert "could not infer" in out


def test_units_named_on_solution(simple_network):
    """A solution can label its species without re-deriving units."""
    conditions = aquakin.SpatialConditions.uniform(1, T=293.15)
    reactor = aquakin.BatchReactor(simple_network, conditions)
    sol = reactor.solve(simple_network.default_concentrations(),
                        t_span=(0.0, 1.0))
    assert sol.units_named("A") == "mol/L"
    # The solution surfaces the time unit too, for unambiguous axis labels.
    assert sol.time_unit == "s"


def test_final_named_and_C_named_many_on_solution(simple_network):
    """final_named / .final report last-point values by name; C_named_many reads
    several trajectories at once -- consistent with C_named."""
    conditions = aquakin.SpatialConditions.uniform(1, T=293.15)
    reactor = aquakin.BatchReactor(simple_network, conditions)
    sol = reactor.solve(simple_network.default_concentrations(),
                        t_span=(0.0, 5.0), t_eval=jnp.linspace(0.0, 5.0, 4))

    # final_named subset: float values equal the last C_named point.
    fn = sol.final_named(["A"])
    assert isinstance(fn["A"], float)
    assert fn["A"] == float(sol.C_named("A")[-1])

    # .final and final_named() cover every species.
    assert set(sol.final) == set(simple_network.species)
    assert sol.final == sol.final_named()

    # C_named_many returns one trajectory per requested name.
    many = sol.C_named_many(["A", "B"])
    assert set(many) == {"A", "B"}
    assert jnp.array_equal(many["A"], sol.C_named("A"))

    # Unknown name raises a hinted KeyError (shared with C_named).
    with pytest.raises(KeyError, match="Did you mean"):
        sol.final_named(["AA"])


def test_solve_time_unit_conversion_is_equivalent(tmp_path):
    """Solving a days network with t in hours gives the same physics, with the
    result times reported back in hours."""
    net = _net_from_yaml(_DAYS_YAML, tmp_path)        # native unit: days
    reactor = aquakin.BatchReactor(net, net.default_conditions())
    C0 = net.default_concentrations()

    sol_d = reactor.solve(C0, t_span=(0.0, 2.0), t_eval=jnp.linspace(0.0, 2.0, 5))
    sol_h = reactor.solve(C0, t_span=(0.0, 48.0), t_eval=jnp.linspace(0.0, 48.0, 5),
                          time_unit="h")             # 48 h == 2 d

    assert sol_d.time_unit == "d"
    assert sol_h.time_unit == "h"
    # Output times come back in the requested unit (the t_eval the caller passed).
    assert jnp.allclose(sol_h.t, jnp.linspace(0.0, 48.0, 5))
    # Same physical times -> identical trajectory.
    assert jnp.allclose(sol_d.C, sol_h.C, rtol=1e-6, atol=1e-8)
    # Passing the native unit explicitly is a no-op.
    sol_d2 = reactor.solve(C0, t_span=(0.0, 2.0), t_eval=jnp.linspace(0.0, 2.0, 5),
                           time_unit="d")
    assert jnp.allclose(sol_d.t, sol_d2.t)


def test_solve_time_unit_unknown_raises(simple_network):
    reactor = aquakin.BatchReactor(simple_network, simple_network.default_conditions())
    with pytest.raises(ValueError, match="Unknown time_unit"):
        reactor.solve(simple_network.default_concentrations(),
                      t_span=(0.0, 1.0), time_unit="fortnight")


def test_solve_time_unit_undeclared_network_raises(tmp_path):
    """If the network's own time unit can't be inferred, conversion can't be
    applied -- raise rather than silently assume."""
    net = _net_from_yaml(_NO_UNITS_YAML, tmp_path)    # time_unit is None
    reactor = aquakin.BatchReactor(net, net.default_conditions())
    with pytest.raises(ValueError, match="could not be inferred"):
        reactor.solve(net.default_concentrations(), t_span=(0.0, 1.0), time_unit="h")


def test_to_latex_smoke(simple_network):
    latex = simple_network.to_latex()
    assert "A_to_B" in latex
    assert "mathrm" in latex["A_to_B"]


def test_concentrations_by_name(simple_network):
    """concentrations() starts from the YAML defaults and overrides named
    species, via a dict and/or kwargs, with no .at[].set() needed."""
    base = simple_network.default_concentrations()
    a = simple_network.species_index["A"]
    b = simple_network.species_index["B"]

    # No overrides -> the defaults unchanged.
    assert jnp.array_equal(simple_network.concentrations(), base)

    # Dict override.
    c = simple_network.concentrations({"A": 5.0})
    assert float(c[a]) == 5.0 and float(c[b]) == float(base[b])

    # kwargs override (identifier-safe name) and dict+kwargs together.
    c2 = simple_network.concentrations({"A": 5.0}, B=2.0)
    assert float(c2[a]) == 5.0 and float(c2[b]) == 2.0


def test_concentrations_zero_base(simple_network):
    """base='zero' starts every species at 0, so only the named ones are set --
    the correct base for a feed composition."""
    a = simple_network.species_index["A"]
    b = simple_network.species_index["B"]

    z = simple_network.concentrations({"A": 5.0}, base="zero")
    assert float(z[a]) == 5.0
    assert float(z[b]) == 0.0          # not the YAML default

    # No overrides + zero base -> all zeros.
    assert jnp.array_equal(
        simple_network.concentrations(base="zero"),
        jnp.zeros_like(simple_network.default_concentrations()),
    )

    # base='defaults' is the (default) old behaviour.
    assert jnp.array_equal(
        simple_network.concentrations({"A": 5.0}, base="defaults"),
        simple_network.concentrations({"A": 5.0}),
    )


def test_concentrations_invalid_base(simple_network):
    with pytest.raises(ValueError, match="base must be"):
        simple_network.concentrations({"A": 5.0}, base="nope")


def test_parameter_values_by_name(simple_network):
    p = simple_network.parameter_values({"A_to_B.k": 0.7})
    assert float(p[simple_network.param_index["A_to_B.k"]]) == 0.7
    # default (no overrides) equals default_parameters
    assert jnp.array_equal(simple_network.parameter_values(),
                           simple_network.default_parameters())


def test_atol_by_name(simple_network):
    atol = simple_network.atol({"B": 1e-15}, default=1e-9)
    assert float(atol[simple_network.species_index["B"]]) == 1e-15
    assert float(atol[simple_network.species_index["A"]]) == 1e-9
    assert atol.shape == (simple_network.n_species,)


def test_override_unknown_name_raises_with_hint(simple_network):
    with pytest.raises(KeyError, match="Unknown species 'AA'"):
        simple_network.concentrations({"AA": 1.0})
    with pytest.raises(KeyError, match="Unknown parameter"):
        simple_network.parameter_values({"A_to_B.kk": 1.0})


def test_override_rejects_non_dict_positional(simple_network):
    with pytest.raises(TypeError, match="must be a dict"):
        simple_network.concentrations(["A", 1.0])


def test_compile_stage_helpers():
    """The compile_network stage helpers behave as a pipeline: parameter index
    is network-level-then-reaction-local in order, and _unresolved_params flags
    only names resolving to neither scope."""
    from aquakin.core.network import _build_param_index, _unresolved_params
    from aquakin.core.parser import parse_rate_expression

    # ASM1 has both network-level (Y_H, ...) and reaction-local params.
    net = aquakin.load_network("asm1")
    # Network-level params come before reaction-local (which are dotted).
    bare = [p for p in net.parameters if "." not in p]
    dotted = [p for p in net.parameters if "." in p]
    if bare and dotted:
        assert net.parameters.index(bare[-1]) < net.parameters.index(dotted[0])

    pidx = {"k": 0, "r1.kf": 1}
    ast = parse_rate_expression("k * kf / nope")
    # 'k' resolves network-level; 'r1.kf' resolves reaction-local; 'nope' doesn't.
    # (order is unspecified -- param_names() is a set -- so compare as sets.)
    assert set(_unresolved_params(ast, "r1", pidx)) == {"nope"}
    assert set(_unresolved_params(ast, "r2", pidx)) == {"kf", "nope"}  # r2.kf absent
