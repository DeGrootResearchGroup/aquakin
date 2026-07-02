"""Every shipped model must render to LaTeX without raising.

``model.to_latex()`` is a documented public API call. A model using a rate
node with no LaTeX renderer (e.g. ``pH_inhibit`` in ADM1) would otherwise raise
``TypeError`` only when a user happens to call it, so render them all here.
"""

from importlib.resources import files

import pytest

import aquakin

_SHIPPED_MODELS = sorted(
    p.name.removesuffix(".yaml")
    for p in files("aquakin.models").iterdir()
    if p.name.endswith(".yaml")
)


@pytest.mark.parametrize("name", _SHIPPED_MODELS)
def test_shipped_model_renders_latex(name):
    model = aquakin.load_model(name)
    rendered = model.to_latex()
    assert set(rendered) == set(model.reaction_names)
    for reaction, latex in rendered.items():
        assert isinstance(latex, str) and latex, f"{name}.{reaction} rendered empty"


def test_adm1_ph_inhibit_renders():
    """ADM1 inlines pH_inhibit into reaction ASTs; regression for the missing
    renderer that made adm1.to_latex() raise."""
    rendered = aquakin.load_model("adm1").to_latex()
    assert any(r"10^{" in latex for latex in rendered.values())
