"""First-class per-species composition metadata + the conservation-check API.

A model may declare each species' content of the conserved quantities
(``COD`` / ``N`` / ``P`` / ``S`` / ``Fe`` / ...) in a ``species[].composition``
block. ``CompiledModel`` carries it as :attr:`species_composition`, exposes it
through :meth:`composition`, and dots it against the stoichiometry in
:meth:`check_conservation` / :meth:`check_nitrogen`. These tests pin:

- the schema parses ``composition:`` and rejects a non-finite / unnamed entry;
- the compiled model carries it and ``composition()`` returns it verbatim;
- ``check_conservation`` flags a deliberately broken coefficient and passes a
  balanced one, and raises a clear error when no composition is available;
- a model that declares no composition falls back to the shipped role-based
  table (the ASM/ADM families), so the API is uniform;
- ``extends:`` inherits a base species' composition.
"""

import textwrap

import numpy as np
import pytest

import aquakin


def _write(tmp_path, body: str, name="net.yaml"):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return p


# A minimal balanced redox toy: S_S + (1/Y) O2 -> X (the O2 demand closes the COD
# balance exactly for the chosen yield). COD content: organic = 1, oxygen = -1.
_TOY = """
model: {{name: toy_balance, version: "1.0"}}
species:
  - {{name: S_S, units: gCOD/m3, default_concentration: 1.0, composition: {{COD: 1.0}}}}
  - {{name: S_O, units: gO2/m3, default_concentration: 8.0, composition: {{COD: -1.0}}}}
  - {{name: X,   units: gCOD/m3, default_concentration: 1.0, composition: {{COD: 1.0}}}}
reactions:
  - name: growth
    rate: "mu * [S_S] * [X]"
    parameters: {{mu: {{value: 1.0}}}}
    stoichiometry: {{S_S: -2.0, S_O: {o2}, X: 1.0}}
"""


def test_schema_parses_and_model_carries_composition(tmp_path):
    net = aquakin.load_model_from_file(_write(tmp_path, _TOY.format(o2=-1.0)))
    assert net.species_composition["S_O"] == {"COD": -1.0}
    comp = net.composition()
    assert comp["S_S"] == {"COD": 1.0}
    assert comp["S_O"] == {"COD": -1.0}
    # composition() returns copies, not the internal dicts.
    comp["S_S"]["COD"] = 999.0
    assert net.species_composition["S_S"]["COD"] == 1.0


def test_check_conservation_passes_balanced_flags_broken(tmp_path):
    # S_S -2, X +1 -> 1 gCOD destroyed, so the O2 demand must be -1 to close COD.
    ok = aquakin.load_model_from_file(_write(tmp_path, _TOY.format(o2=-1.0)))
    assert ok.check_conservation(tol=1e-6) == []
    # A wrong O2 coefficient breaks the COD balance and is flagged.
    bad = aquakin.load_model_from_file(_write(tmp_path, _TOY.format(o2=-0.5), name="bad.yaml"))
    viol = bad.check_conservation(tol=1e-6)
    assert [(r, q) for r, q, _ in viol] == [("growth", "COD")]
    assert abs(viol[0][2]) == pytest.approx(0.5, abs=1e-9)


def test_check_conservation_without_metadata_raises(tmp_path):
    # ozone_bromate declares no composition and has no shipped role-based table.
    net = aquakin.load_model("ozone_bromate")
    assert net.species_composition == {}
    assert net.composition() == {}
    with pytest.raises(ValueError, match="no composition metadata"):
        net.check_conservation()


def test_composition_falls_back_to_shipped_table_for_asm():
    """A model with no YAML composition delegates to the shipped role-based
    table, so check_conservation works uniformly across families."""
    asm1 = aquakin.load_model("asm1")
    assert asm1.species_composition == {}  # none declared in YAML
    comp = asm1.composition()
    assert comp, "expected the shipped ASM1 role-based composition table"
    # ASM1 conserves COD through the Gujer matrix (except the single
    # denitrification reaction, which the electron convention handles -- here we
    # just assert the fallback produces a usable table).
    assert comp["SO"] == {"COD": -1.0}


@pytest.mark.parametrize(
    "name",
    [
        "asm1",
        "asm1_ammonia_limitation",
        "asm2d",
        "asm2d_tud",
        "asm3",
        "asm3_biop",
        "asm3_2step",
        "asm3_2step_n2o",
        "asm3_2step_anammox",
        "asm3_2step_comammox",
        "adm1",
    ],
)
def test_shipped_composition_table_warns_for_no_species(recwarn, name):
    """Every shipped species is covered by a role (or the known-inert set), so
    the shipped table must not warn -- the guarantee the unmapped-species warning
    depends on to stay signal, not noise."""
    aquakin.load_model(name).composition()
    unmapped = [w for w in recwarn.list if "not recognised" in str(w.message)]
    assert not unmapped, [str(w.message) for w in unmapped]


class _StubModel:
    """A minimal ``CompiledModel`` stand-in for driving the role-based builders
    directly with an arbitrary species list (no parameters, so every ``i*``
    fraction defaults to 0)."""

    def __init__(self, name, species):
        self.name = name
        self.species = species
        self.param_index: dict = {}

    def default_parameters(self):
        return np.zeros(0)


def test_unmapped_species_warns_instead_of_silent_empty_content():
    """A species the role sets do not recognise must warn (and be listed), not
    silently get zero COD/N/P -- which would make a conservation check validate
    against wrong reference data. Recognised no-content species (SALK) stay
    silent."""
    from aquakin.utils.composition import _asm_composition

    model = _StubModel("asm1_variant", ["XB_H", "SNH", "SALK", "XWEIRD", "SNEW_N"])
    with pytest.warns(UserWarning, match="not recognised"):
        comp = _asm_composition(model)
    # The recognised species keep their content; the unknowns get empty content
    # (the pre-existing behaviour) -- but now loudly, not silently.
    assert comp["XWEIRD"] == {} and comp["SNEW_N"] == {}
    assert comp["SALK"] == {}  # a known-inert carrier -- did not trigger the warning
    assert comp["SNH"] == {"N": 1.0}


def test_unmapped_adm1_species_warns():
    """The ADM1 builder is loud about an unrecognised species too."""
    from aquakin.utils.composition import _adm1_composition

    model = _StubModel("adm1", ["S_su", "S_IN", "S_cat", "X_NEW"])
    with pytest.warns(UserWarning, match=r"X_NEW"):
        comp = _adm1_composition(model)
    assert comp["X_NEW"] == {}
    assert comp["S_cat"] == {}  # known charge-only state -- silent


def test_composition_rejects_non_finite(tmp_path):
    body = _TOY.format(o2=-1.0).replace("composition: {COD: -1.0}", "composition: {COD: .inf}")
    with pytest.raises(ValueError, match="must be finite"):
        aquakin.load_model_from_file(_write(tmp_path, body))


def test_extends_inherits_base_composition(tmp_path):
    base = _write(tmp_path, _TOY.format(o2=-1.0), name="base.yaml")
    # A derived model that overrides only the rate keeps the base composition.
    derived = _write(
        tmp_path,
        f"""
        model: {{name: toy_child, extends: {base}}}
        reactions:
          - name: growth
            rate: "mu * [S_S]"
        """,
        name="child.yaml",
    )
    child = aquakin.load_model_from_file(derived)
    assert child.composition()["S_O"] == {"COD": -1.0}
    assert child.check_conservation(tol=1e-6) == []
