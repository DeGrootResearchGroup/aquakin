"""The default concentrations of the shipped models describe a real state.

``model.default_concentrations()`` is what the guide's first solves start
from, so a model whose defaults are placeholders gives a meaningless first run.
"""

import pathlib

import pytest

import aquakin

MODELS = sorted(p.stem for p in (pathlib.Path(aquakin.__file__).parent / "models").glob("*.yaml"))

# Each model's TSS state and, per particulate, the parameter converting it to TSS.
TSS_FACTORS = {
    "asm3": ("XSS", {"XI": "iSS_XI", "XS": "iSS_XS", "XH": "iSS_BM", "XA": "iSS_BM",
                     "XSTO": "iSS_STO"}),
    "asm2d": ("XTSS", {"XI": "iTSS_XI", "XS": "iTSS_XS", "XH": "iTSS_BM", "XPAO": "iTSS_BM",
                       "XAUT": "iTSS_BM", "XPP": "iTSS_XPP", "XPHA": "iTSS_XPHA"}),
    "asm2d_tud": ("XTSS", {"XI": "iTSS_XI", "XS": "iTSS_XS", "XH": "iTSS_BM",
                           "XPAO": "iTSS_BM", "XA": "iTSS_BM", "XPP": "iTSS_PP",
                           "XPHA": "iTSS_PHA", "XGLY": "iTSS_GLY"}),
    "asm3_biop": ("XTSS", {"XI": "iTSS_XI", "XS": "iTSS_XS", "XH": "iTSS_BM",
                           "XPAO": "iTSS_BM", "XA": "iTSS_BM", "XPP": "iTSS_XPP",
                           "XPHA": "iTSS_XSTO", "XSTO": "iTSS_XSTO"}),
}


@pytest.mark.parametrize("name", MODELS)
def test_defaults_are_not_placeholders(name):
    """No more than a third of a model's default concentrations is exactly 1.0,
    the value a model imported without its reference state carries for every
    species."""
    c = aquakin.load_model(name).default_concentrations()
    assert int((c == 1.0).sum()) <= len(c) // 3


@pytest.mark.parametrize("name", sorted(TSS_FACTORS))
def test_default_tss_matches_the_default_particulates(name):
    """The default suspended solids are the TSS of the default particulates,
    through the model's own COD- and P-to-TSS factors."""
    model = aquakin.load_model(name)
    tss, factors = TSS_FACTORS[name]
    c = dict(zip(model.species, map(float, model.default_concentrations())))
    p = dict(zip(model.parameters, map(float, model.default_parameters())))
    expected = sum(c[s] * p[f] for s, f in factors.items())
    assert c[tss] == pytest.approx(expected, abs=0.05)
    assert c[tss] > 500.0  # an activated-sludge mixed liquor, not a trace
