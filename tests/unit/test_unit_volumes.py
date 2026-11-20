"""Every unit with a volume refuses one that is not a positive finite number: a
zero or negative volume divides its mass balance, and a steady-state solve of
such a plant does not end."""

import pytest

import aquakin
from aquakin.plant import ChlorineContactUnit, UVUnit
from aquakin.plant._constants import check_positive
from aquakin.plant.catalog import check_unit


@pytest.mark.parametrize("kind", ["cstr", "mbr", "digester", "primary_clarifier", "storage"])
@pytest.mark.parametrize("volume", [-1.0, 0.0, "inf"])
def test_a_unit_refuses_a_volume_that_is_not_positive(kind, volume):
    model = "adm1" if kind == "digester" else "asm1"
    reply = check_unit(kind, "u", {"volume": volume, "model": model})
    assert not reply["ok"] and "u: volume must be a positive number" in reply["error"]


def test_disinfection_units_refuse_it_too():
    asm1 = aquakin.load_model("asm1")
    with pytest.raises(ValueError, match="uv: volume must be a positive number"):
        UVUnit("uv", asm1, volume=0.0, intensity=10.0, d10=5.0)
    with pytest.raises(ValueError, match="cl: volume must be a positive number"):
        ChlorineContactUnit("cl", asm1, volume=-2.0, dose=5.0, ct_per_log=1.0)


def test_the_check_names_text_that_is_no_number():
    with pytest.raises(ValueError, match="t: volume must be a positive number; got 'big'"):
        check_positive("t", "volume", "big")
    check_positive("t", "volume", 1e-6)
