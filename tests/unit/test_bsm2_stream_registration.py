"""Fast unit tests for the BSM2 semantic-stream registration.

``_register_bsm2_streams`` is the pure metadata-bookkeeping block factored out of
``build_bsm2``: it resolves the option-dependent influent/effluent endpoints and
registers the ``plant.stream(sol, name)`` shortcuts, independent of the unit
wiring. So it is checked here against a lightweight stub plant -- no unit
construction, no solve -- covering the endpoint precedence (delay > bypass >
front) and the full set of registered names.
"""

from aquakin.plant.bsm.bsm2 import _register_bsm2_streams


class _StubPlant:
    """Minimal stand-in exposing only what the registration touches."""

    def __init__(self):
        self.named_streams = {}
        self.influent_endpoint = None
        self.effluent_endpoint = None

    def register_stream(self, name, endpoint):
        self.named_streams[name] = endpoint
        return self


# The names every BSM2 plant registers, regardless of options.
_EXPECTED_NAMES = {
    "effluent",
    "internal_recycle",
    "ras",
    "wastage",
    "primary_effluent",
    "primary_sludge",
    "thickener_overflow",
    "reject",
    "dewatering_reject",
    "disposal_sludge",
}


def test_default_endpoints_and_registered_names():
    p = _StubPlant()
    _register_bsm2_streams(p, influent_bypass=False, use_delay=False)
    assert p.influent_endpoint == "front_mix.fresh"
    assert p.effluent_endpoint == "settler.overflow"
    assert set(p.named_streams) == _EXPECTED_NAMES
    # "effluent" tracks the resolved effluent_endpoint.
    assert p.named_streams["effluent"] == "settler.overflow"
    # A couple of the fixed shortcuts.
    assert p.named_streams["ras"] == "underflow_split.ras"
    assert p.named_streams["reject"] == "reject_mix.out"


def test_bypass_moves_influent_and_effluent_endpoints():
    p = _StubPlant()
    _register_bsm2_streams(p, influent_bypass=True, use_delay=False)
    assert p.influent_endpoint == "bypass_split.in"
    assert p.effluent_endpoint == "effluent_mix.out"
    assert p.named_streams["effluent"] == "effluent_mix.out"


def test_delay_takes_precedence_over_bypass_for_influent():
    # The hydraulic delay is front-most, so it wins the influent endpoint even
    # when bypass is also on; the effluent endpoint still follows bypass.
    p = _StubPlant()
    _register_bsm2_streams(p, influent_bypass=True, use_delay=True)
    assert p.influent_endpoint == "influent_delay.in"
    assert p.effluent_endpoint == "effluent_mix.out"


def test_delay_without_bypass():
    p = _StubPlant()
    _register_bsm2_streams(p, influent_bypass=False, use_delay=True)
    assert p.influent_endpoint == "influent_delay.in"
    assert p.effluent_endpoint == "settler.overflow"
    assert p.named_streams["effluent"] == "settler.overflow"
