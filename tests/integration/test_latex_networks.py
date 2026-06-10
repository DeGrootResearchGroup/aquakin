"""Every shipped network must render to LaTeX without raising.

``network.to_latex()`` is a documented public API call. A network using a rate
node with no LaTeX renderer (e.g. ``pH_inhibit`` in ADM1) would otherwise raise
``TypeError`` only when a user happens to call it, so render them all here.
"""

from importlib.resources import files

import pytest

import aquakin

_SHIPPED_NETWORKS = sorted(
    p.name.removesuffix(".yaml")
    for p in files("aquakin.networks").iterdir()
    if p.name.endswith(".yaml")
)


@pytest.mark.parametrize("name", _SHIPPED_NETWORKS)
def test_shipped_network_renders_latex(name):
    network = aquakin.load_network(name)
    rendered = network.to_latex()
    assert set(rendered) == set(network.reaction_names)
    for reaction, latex in rendered.items():
        assert isinstance(latex, str) and latex, f"{name}.{reaction} rendered empty"


def test_adm1_ph_inhibit_renders():
    """ADM1 inlines pH_inhibit into reaction ASTs; regression for the missing
    renderer that made adm1.to_latex() raise."""
    rendered = aquakin.load_network("adm1").to_latex()
    assert any(r"10^{" in latex for latex in rendered.values())
