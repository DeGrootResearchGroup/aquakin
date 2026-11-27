"""Direct kernel tests for the effluent / cost metric functions.

These exercise the public metric kernels on hand-built concentration vectors
with closed-form expected values -- no plant solve. They were previously
covered only transitively (``derived_BOD`` / ``derived_TKN`` via
``effluent_averages`` and the EQI; ``operational_cost_index`` via
``evaluate_bsm1``), so a regression in a coefficient could slip through.
"""

import jax.numpy as jnp
import pytest

import aquakin


@pytest.fixture(scope="module")
def asm1():
    return aquakin.load_model("asm1")


def test_derived_BOD_matches_closed_form(asm1):
    # BOD5 proxy = 0.25 * (SS + XS + (1 - f_P) * (XB_H + XB_A)), f_P = 0.08.
    C = asm1.concentrations(
        {"SS": 10.0, "XS": 20.0, "XB_H": 100.0, "XB_A": 50.0}, base="zero"
    )
    expected = 0.25 * (10.0 + 20.0 + (1.0 - 0.08) * (100.0 + 50.0))  # = 42.0
    assert float(aquakin.derived_BOD(C, asm1)) == pytest.approx(expected)


def test_derived_TKN_matches_closed_form(asm1):
    # TKN = SNH + SND + XND + i_XB*(XB_H + XB_A) + i_XP*(XP + XI),
    # with i_XB = 0.086, i_XP = 0.06.
    C = asm1.concentrations(
        {"SNH": 5.0, "SND": 2.0, "XND": 3.0, "XB_H": 100.0, "XB_A": 50.0,
         "XP": 30.0, "XI": 40.0},
        base="zero",
    )
    expected = (5.0 + 2.0 + 3.0
                + 0.086 * (100.0 + 50.0)
                + 0.06 * (30.0 + 40.0))            # = 27.1
    assert float(aquakin.derived_TKN(C, asm1)) == pytest.approx(expected)


def test_derived_kernels_vectorize_over_a_trajectory(asm1):
    # The kernels index C[..., i], so a 2-D (n_t, n_species) trajectory maps to a
    # leading-axis vector with no rank branch.
    row = asm1.concentrations(
        {"SS": 10.0, "XS": 20.0, "XB_H": 100.0, "XB_A": 50.0}, base="zero"
    )
    traj = jnp.stack([row, 2.0 * row])             # (2, n_species)
    bod = aquakin.derived_BOD(traj, asm1)
    assert bod.shape == (2,)
    # linear in C, so doubling the state doubles BOD
    assert float(bod[1]) == pytest.approx(2.0 * float(bod[0]))


def test_operational_cost_index_bsm1_form():
    # BSM1 OCI = aeration + pumping + mixing + 5 * sludge_production.
    assert aquakin.operational_cost_index(10.0, 5.0, 2.0) == pytest.approx(25.0)
    # weight is exactly 5 on the sludge term
    base = aquakin.operational_cost_index(10.0, 5.0, 0.0)
    assert aquakin.operational_cost_index(10.0, 5.0, 1.0) - base == pytest.approx(5.0)
    # the optional mixing term adds directly (default 0 keeps the Copp 2002 form)
    assert aquakin.operational_cost_index(10.0, 5.0, 2.0, mixing=3.0) == pytest.approx(28.0)
    assert isinstance(aquakin.operational_cost_index(1.0, 1.0, 1.0), float)


# ----- COD and BOD read from the model (not ASM1's species) -----------------
#
# The COD species and the biodegradable fractions come from each model; the
# expected values here are written out by hand from the models' published
# matrices (Henze et al. 2000), with the inert fractions read by name.

_COD_SPECIES = {
    "asm2d": ("SF", "SA", "SI", "XI", "XS", "XH", "XPAO", "XPHA", "XAUT"),
    "asm3": ("SS", "SI", "XI", "XS", "XH", "XSTO", "XA"),
}


def _distinct(model):
    """A state with a different concentration in every species."""
    return jnp.arange(1.0, model.n_species + 1.0) * 1.7


def _hand_bod(name, C, p):
    """0.25 x the biodegradable COD: substrates and storage whole, hydrolysed
    XS less its soluble-inert share fSI, biomass less its decay-inert share fXI."""
    index = aquakin.load_model(name).species_index
    c = {s: float(C[index[s]]) for s in _COD_SPECIES[name]}
    if name == "asm2d":
        whole = c["SF"] + c["SA"] + c["XPHA"]
        biomass = c["XH"] + c["XPAO"] + c["XAUT"]
    else:
        whole = c["SS"] + c["XSTO"]
        biomass = c["XH"] + c["XA"]
    return 0.25 * (whole + (1.0 - p["fSI"]) * c["XS"] + (1.0 - p["fXI"]) * biomass)


@pytest.mark.parametrize("name", sorted(_COD_SPECIES))
def test_derived_COD_counts_every_cod_species_of_the_model(name):
    """On ASM2d / ASM3 the COD is the sum of that model's COD species, not of the
    ASM1-named ones it happens to share (SI, XI, XS)."""
    model = aquakin.load_model(name)
    C = _distinct(model)
    expected = sum(float(C[model.species_index[s]]) for s in _COD_SPECIES[name])
    assert float(aquakin.derived_COD(C, model)) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("name", sorted(_COD_SPECIES))
def test_derived_BOD_weighs_the_models_biodegradable_cod(name):
    """BOD on ASM2d / ASM3 (which have no SS / XB_H) is 0.25 x the biodegradable
    COD, and it follows the inert fractions of the parameters given."""
    model = aquakin.load_model(name)
    C = _distinct(model)
    defaults = model.default_parameters()
    p = {k: float(defaults[model.param_index[k]]) for k in ("fSI", "fXI")}
    assert float(aquakin.derived_BOD(C, model)) == pytest.approx(
        _hand_bod(name, C, p), rel=1e-12
    )
    p2 = {"fSI": 0.05, "fXI": 0.15}
    params = defaults
    for k, v in p2.items():
        params = params.at[model.param_index[k]].set(v)
    assert float(aquakin.derived_BOD(C, model, params=params)) == pytest.approx(
        _hand_bod(name, C, p2), rel=1e-12
    )


def test_derived_BOD_f_P_needs_an_f_P_parameter():
    """``f_P`` overrides ASM1's parameter; on a model without one it raises and
    says to pass the parameters instead, which works."""
    asm3 = aquakin.load_model("asm3")
    C = _distinct(asm3)
    with pytest.raises(ValueError, match=r"has no f_P parameter.*pass params="):
        aquakin.derived_BOD(C, asm3, f_P=0.08)
    params = asm3.default_parameters().at[asm3.param_index["fXI"]].set(0.08)
    p = {"fSI": float(params[asm3.param_index["fSI"]]), "fXI": 0.08}
    assert float(aquakin.derived_BOD(C, asm3, params=params)) == pytest.approx(
        _hand_bod("asm3", C, p), rel=1e-12
    )


def test_derived_BOD_f_P_overrides_the_asm1_parameter(asm1):
    C = asm1.concentrations({"SS": 10.0, "XS": 20.0, "XB_H": 100.0, "XB_A": 50.0}, base="zero")
    expected = 0.25 * (10.0 + 20.0 + (1.0 - 0.2) * 150.0)
    assert float(aquakin.derived_BOD(C, asm1, f_P=0.2)) == pytest.approx(expected, rel=1e-12)


# XB decays by two routes leaving different inert shares, so its biodegradable
# COD is not one number. No composition: the COD species come from the units.
_TWO_ROUTES = """
model: {name: bod_two_routes, version: "1.0", description: "test", references: []}
species:
  - {name: S,  units: "g_COD/m3", default_concentration: 1.0}
  - {name: XB, units: "g_COD/m3", default_concentration: 1.0}
  - {name: XI, units: "g_COD/m3", default_concentration: 1.0}
  - {name: SN, units: "g_N/m3",   default_concentration: 1.0}
conditions: [{name: T, default: 293.15}]
reactions:
  - name: growth
    rate: "k * [S]"
    parameters: {k: {value: 1.0}}
    stoichiometry: {S: -1, XB: 1}
  - name: decay_a
    rate: "k * [XB]"
    parameters: {k: {value: 1.0}}
    stoichiometry: {XB: -1, XI: 0.1, S: 0.9}
  - name: decay_b
    rate: "k * [XB]"
    parameters: {k: {value: 1.0}}
    stoichiometry: {XB: -1, XI: 0.2, S: 0.8}
"""


def _model_file(tmp_path, text):
    path = tmp_path / "m.yaml"
    path.write_text(text)
    return aquakin.load_model_from_file(path)


def test_undetermined_bod_raises_and_cod_still_works(tmp_path):
    """A species whose consuming reactions leave different inert shares has no
    single biodegradable fraction: BOD raises and names COD, whose value (from
    the units: S + XB + XI, SN being nitrogen) is then exact."""
    model = _model_file(tmp_path, _TWO_ROUTES)
    C = jnp.asarray([1.0, 2.0, 4.0, 8.0])
    with pytest.raises(ValueError, match=r"BOD cannot be determined; use substrate=\"COD\""):
        aquakin.derived_BOD(C, model)
    assert float(aquakin.derived_COD(C, model)) == pytest.approx(7.0, rel=1e-12)


def test_cod_needs_readable_units_or_a_composition(tmp_path):
    model = _model_file(tmp_path, _TWO_ROUTES.replace('"g_N/m3"', '"furlongs"'))
    with pytest.raises(ValueError, match=r"units of \['SN'\] do not say whether they carry COD"):
        aquakin.derived_COD(jnp.ones(4), model)


def test_cod_needs_a_cod_species(tmp_path):
    model = _model_file(tmp_path, _TWO_ROUTES.replace('"g_COD/m3"', '"g_N/m3"'))
    with pytest.raises(ValueError, match="has no species that carries COD"):
        aquakin.derived_COD(jnp.ones(4), model)


# ----- TKN, NOx, TSS and the effluent averages read from the model ----------
#
# The expected values are written out by hand from each model's published
# composition (Henze et al. 2000): its N fractions and TSS factors, read by name.


def _p(model, *names):
    d = model.default_parameters()
    return {k: float(d[model.param_index[k]]) for k in names}


def _c(model, C, *names):
    return {s: float(C[model.species_index[s]]) for s in names}


def _hand_tkn(name, C, p):
    """Ammonium plus each organic pool's N: iN_BM on biomass, iN_* elsewhere."""
    model = aquakin.load_model(name)
    if name == "asm2d":
        c = _c(model, C, "SNH4", "SF", "SI", "XI", "XS", "XH", "XPAO", "XAUT")
        soluble = p["iN_SF"] * c["SF"]
        biomass = c["XH"] + c["XPAO"] + c["XAUT"]
    else:
        c = _c(model, C, "SNH4", "SS", "SI", "XI", "XS", "XH", "XA")
        soluble = p["iN_SS"] * c["SS"]
        biomass = c["XH"] + c["XA"]
    return (c["SNH4"] + soluble + p["iN_SI"] * c["SI"] + p["iN_XI"] * c["XI"]
            + p["iN_XS"] * c["XS"] + p["iN_BM"] * biomass)


@pytest.mark.parametrize("name", ["asm2d", "asm3"])
def test_derived_TKN_counts_the_models_nitrogen(name):
    """TKN on ASM2d / ASM3 (no SND, XND or XB_H) is ammonium plus the organic N
    of every pool, and it follows the N fractions of the parameters given."""
    model = aquakin.load_model(name)
    C = _distinct(model)
    names = ("iN_BM", "iN_SI", "iN_XI", "iN_XS", "iN_SF" if name == "asm2d" else "iN_SS")
    p = _p(model, *names)
    assert float(aquakin.derived_TKN(C, model)) == pytest.approx(
        _hand_tkn(name, C, p), rel=1e-12
    )
    params = model.default_parameters().at[model.param_index["iN_BM"]].set(0.1)
    assert float(aquakin.derived_TKN(C, model, params=params)) == pytest.approx(
        _hand_tkn(name, C, {**p, "iN_BM": 0.1}), rel=1e-12
    )


def test_derived_TKN_i_XB_needs_an_i_XB_parameter():
    """``i_XB`` overrides ASM1's parameter; on ASM2d it raises and says to pass
    the parameters, which works."""
    asm2d = aquakin.load_model("asm2d")
    C = _distinct(asm2d)
    with pytest.raises(ValueError, match=r"has no parameter \['i_XB'\].*pass params="):
        aquakin.derived_TKN(C, asm2d, i_XB=0.08)
    params = asm2d.default_parameters().at[asm2d.param_index["iN_BM"]].set(0.08)
    p = _p(asm2d, "iN_SF", "iN_SI", "iN_XI", "iN_XS")
    assert float(aquakin.derived_TKN(C, asm2d, params=params)) == pytest.approx(
        _hand_tkn("asm2d", C, {**p, "iN_BM": 0.08}), rel=1e-12
    )


def test_derived_TKN_i_XB_overrides_the_asm1_parameter(asm1):
    C = asm1.concentrations({"SNH": 5.0, "XB_H": 100.0, "XB_A": 50.0, "XI": 40.0}, base="zero")
    expected = 5.0 + 0.08 * 150.0 + 0.05 * 40.0
    assert float(aquakin.derived_TKN(C, asm1, i_XB=0.08, i_XP=0.05)) == pytest.approx(
        expected, rel=1e-12
    )


@pytest.mark.parametrize(
    ("name", "nox"),
    [("asm1", ("SNO",)), ("asm2d", ("SNO3",)), ("asm3", ("SNOX",)),
     # nitrite and nitrate; nitric oxide (SNO here), N2O and N2 are not NOx
     ("asm3_2step_n2o", ("SNO2", "SNO3"))],
)
def test_derived_NOx_is_nitrate_plus_nitrite(name, nox):
    model = aquakin.load_model(name)
    C = _distinct(model)
    expected = sum(_c(model, C, *nox).values())
    assert float(aquakin.derived_NOx(C, model)) == pytest.approx(expected, rel=1e-12)


def _hand_tss_asm2d(model, C):
    p = _p(model, "iTSS_XI", "iTSS_XS", "iTSS_BM", "iTSS_XPP", "iTSS_XPHA")
    c = _c(model, C, "XI", "XS", "XH", "XPAO", "XAUT", "XPP", "XPHA")
    return (p["iTSS_XI"] * c["XI"] + p["iTSS_XS"] * c["XS"]
            + p["iTSS_BM"] * (c["XH"] + c["XPAO"] + c["XAUT"])
            + p["iTSS_XPP"] * c["XPP"] + p["iTSS_XPHA"] * c["XPHA"])


def test_derived_TSS_weighs_asm2d_solids_by_their_tss_factors():
    """ASM2d's TSS is each particulate through its iTSS factor, the metal
    precipitates at 1 (already g TSS) and the XTSS state itself at 0."""
    asm2d = aquakin.load_model("asm2d")
    C = _distinct(asm2d)
    expected = _hand_tss_asm2d(asm2d, C) + sum(_c(asm2d, C, "XMeOH", "XMeP").values())
    assert float(aquakin.derived_TSS(C, asm2d)) == pytest.approx(expected, rel=1e-12)


def test_derived_TSS_counts_the_iron_precipitates_by_their_molar_mass():
    """asm2d_chemp's strengite and ferrihydrite (mol/m3) add their molar mass
    to the suspended solids: 150.82 and 106.87 g TSS/mol."""
    chemp = aquakin.load_model("asm2d_chemp")
    C = _distinct(chemp)
    iron = _c(chemp, C, "X_FePO4", "X_FeOH3")
    expected = (_hand_tss_asm2d(chemp, C)
                + 150.82 * iron["X_FePO4"] + 106.87 * iron["X_FeOH3"])
    assert float(aquakin.derived_TSS(C, chemp)) == pytest.approx(expected, rel=1e-12)


def test_asm2d_chemp_precipitation_keeps_XTSS_in_step_with_its_solids():
    """Each precipitation reaction changes XTSS by the TSS of the solid it forms,
    so the weighted particulates stay equal to the XTSS state."""
    chemp = aquakin.load_model("asm2d_chemp")
    S = chemp.compute_stoich(chemp.default_parameters())
    col = chemp.species_index
    for reaction, solid, mw in (("FePO4_precipitation", "X_FePO4", 150.82),
                                ("FeOH3_precipitation", "X_FeOH3", 106.87)):
        r = chemp.reaction_names.index(reaction)
        assert float(S[r, col["XTSS"]]) == pytest.approx(mw * float(S[r, col[solid]]), rel=1e-12)


def test_effluent_averages_and_eqi_on_asm2d():
    """The averages and the EQI of an ASM2d effluent: ammonium is SNH4, NOx is
    SNO3, and TKN / TSS / COD / BOD are the model's own."""
    asm2d = aquakin.load_model("asm2d")
    row = _distinct(asm2d)
    C = jnp.stack([row, row])          # constant over the window
    t, Q = jnp.asarray([0.0, 1.0]), jnp.asarray([1000.0, 1000.0])
    avg = aquakin.effluent_averages(t, C, Q, asm2d)
    p = _p(asm2d, "iN_BM", "iN_SF", "iN_SI", "iN_XI", "iN_XS")
    c = _c(asm2d, row, "SNH4", "SNO3")
    tss = float(aquakin.derived_TSS(row, asm2d))
    cod = sum(_c(asm2d, row, *_COD_SPECIES["asm2d"]).values())
    bod = _hand_bod("asm2d", row, _p(asm2d, "fSI", "fXI"))
    tkn = _hand_tkn("asm2d", row, p)
    expected = {"TSS": tss, "COD": cod, "BOD": bod, "TKN": tkn,
                "SNH": c["SNH4"], "SNO": c["SNO3"]}
    for key, value in expected.items():
        # the flow-weighted average divides by sum(Q dt) + EPS_Q
        assert avg[key] == pytest.approx(value, rel=1e-9), key
    eqi = 1000.0 * (2 * tss + cod + 2 * bod + 30 * tkn + 10 * c["SNO3"]) * 1e-3
    assert aquakin.effluent_quality_index(t, C, Q, asm2d) == pytest.approx(eqi, rel=1e-12)


def test_derived_TSS_and_ammonium_need_a_model_that_has_them(tmp_path):
    from aquakin.plant.measures import ammonium_species

    model = _model_file(tmp_path, _TWO_ROUTES)
    C = jnp.ones(model.n_species)
    with pytest.raises(ValueError, match=r"has no TSS state.*cannot be derived"):
        aquakin.derived_TSS(C, model)
    with pytest.raises(KeyError, match=r"has no ammonium species.*read its nitrogen as TKN"):
        ammonium_species(model)
