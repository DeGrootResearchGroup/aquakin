"""Influent characterization + CSV column_map (issue #136).

``fractionate`` / ``characterize_influent`` map influent aggregate measurements
(total COD, TKN, ammonia, alkalinity, optional filtered/flocculated COD) to the
ASM1 state vector, following the SUMO Sumo1 raw-influent fractionation reduced to
ASM1; ``read_influent_csv(column_map=...)`` does it per row for an
arbitrary-header CSV. Checks the SUMO worked example, COD/TKN conservation, the
fractions-only and array paths, and the column_map loader.
"""
import numpy as np
import pytest

import aquakin
from aquakin import (
    InfluentFractions,
    characterize_influent,
    fractionate,
)
from aquakin.plant.influent import _influent_from_column_map

_COD_STATES = ("SI", "SS", "XI", "XS", "XB_H", "XB_A", "XP")


@pytest.fixture(scope="module")
def asm1():
    return aquakin.load_network("asm1")


# --- fractionate against the SUMO Sumo1 raw-influent worked example ----------

def test_fractionate_matches_sumo_example():
    # the spreadsheet's example: TCOD 420, filtered 170, flocculated 85,
    # soluble-inert 20, TKN 34.4, ammonia 24, alkalinity 330.
    s = fractionate(total_cod=420.0, tkn=34.4, ammonia=24.0, alkalinity=330.0,
                    filtered_cod=170.0, flocculated_filtered_cod=85.0,
                    soluble_inert_cod=20.0)
    assert s["SI"] == pytest.approx(20.0)
    assert s["SS"] == pytest.approx(65.0)
    assert s["XI"] == pytest.approx(75.8)
    assert s["XS"] == pytest.approx(234.0)
    assert s["XB_H"] == pytest.approx(21.0)
    assert s["XP"] == pytest.approx(4.2)
    assert s["XB_A"] == 0.0 and s["SO"] == 0.0
    assert s["SNH"] == pytest.approx(24.0)
    assert s["SALK"] == pytest.approx(6.6)        # 330 / 50 mg CaCO3 -> mol/m3


def test_fractionate_conserves_total_cod():
    s = fractionate(total_cod=420.0, tkn=34.4)
    assert sum(s[k] for k in _COD_STATES) == pytest.approx(420.0)


def test_fractionate_closes_tkn_balance():
    f = InfluentFractions()
    s = fractionate(total_cod=420.0, tkn=34.4, ammonia=24.0, fractions=f)
    # ASM1 TKN = SNH + SND + XND + i_XB*XB_H + i_XP*XP  (excludes nitrate)
    tkn = (s["SNH"] + s["SND"] + s["XND"]
           + f.iN_xb * s["XB_H"] + f.iN_xp * s["XP"])
    assert tkn == pytest.approx(34.4)


def test_fractions_only_path_is_reasonable():
    # no filtered-COD measurements -> the default fractions drive the split
    s = fractionate(total_cod=400.0, tkn=40.0)
    assert sum(s[k] for k in _COD_STATES) == pytest.approx(400.0)
    assert s["SS"] > 0 and s["XS"] > s["SS"]      # particulate-dominant municipal


def test_fractionate_is_vectorised_per_row():
    cod = np.array([420.0, 500.0, 380.0])
    tkn = np.array([34.4, 40.0, 30.0])
    s = fractionate(total_cod=cod, tkn=tkn)
    assert s["SS"].shape == (3,)
    # each row equals the scalar call
    for i in range(3):
        si = fractionate(total_cod=float(cod[i]), tkn=float(tkn[i]))
        assert float(s["SS"][i]) == pytest.approx(si["SS"])


def test_fractionate_clamps_negative_states():
    # a tiny COD with a large biomass fraction must not go negative
    s = fractionate(total_cod=50.0, tkn=5.0,
                    fractions=InfluentFractions(f_xu=0.5, f_oho=0.5))
    assert all(np.all(np.asarray(v) >= 0.0) for v in s.values())


# --- characterize_influent --------------------------------------------------

def test_characterize_influent_builds_series(asm1):
    inf = characterize_influent(asm1, flow=24000.0, total_cod=420.0, tkn=34.4,
                                ammonia=24.0, alkalinity=330.0,
                                filtered_cod=170.0, flocculated_filtered_cod=85.0,
                                soluble_inert_cod=20.0, T=288.0)
    si = asm1.species_index
    assert float(inf.Q[0]) == 24000.0
    assert float(inf.C[0, si["SS"]]) == pytest.approx(65.0)
    assert float(inf.C[0, si["XB_A"]]) == 0.0      # no autotrophs in raw influent
    assert float(inf.T[0]) == 288.0


def test_characterize_requires_asm1_states():
    ozone = aquakin.load_network("ozone_bromate")
    with pytest.raises(ValueError, match="ASM1 state"):
        characterize_influent(ozone, flow=1.0, total_cod=1.0, tkn=1.0)


# --- read_influent_csv(column_map=...) --------------------------------------

_LAB_CSV = """day,flow_m3d,COD,TKN,NH4-N,Alk,NOx,temp_C
0,24000,420,34.4,24,330,0,11.8
0.5,28000,500,40,28,350,1.5,12.1
1.0,22000,380,30,21,310,0,12.5
"""
_MAP = {"t": "day", "Q": "flow_m3d", "T": "temp_C", "total_cod": "COD",
        "tkn": "TKN", "ammonia": "NH4-N", "alkalinity": "Alk", "SNO": "NOx"}


def test_column_map_fractionates_each_row(asm1):
    inf = _influent_from_column_map(_LAB_CSV, asm1, _MAP, None, None, "<t>")
    si = asm1.species_index
    assert inf.t.shape[0] == 3
    assert float(inf.C[0, si["SS"]]) == pytest.approx(
        float(fractionate(total_cod=420.0, tkn=34.4, ammonia=24.0)["SS"]))
    # the directly-mapped NOx column overrides the (zero) fractionated SNO
    assert float(inf.C[1, si["SNO"]]) == 1.5
    assert float(inf.C[0, si["SNO"]]) == 0.0


def test_column_map_requires_total_cod_and_tkn(asm1):
    bad = {"t": "day", "Q": "flow_m3d", "ammonia": "NH4-N"}   # aggregate, no COD/TKN
    with pytest.raises(ValueError, match="total_cod"):
        _influent_from_column_map(_LAB_CSV, asm1, bad, None, None, "<t>")


def test_column_map_unknown_header_raises(asm1):
    bad = {"t": "day", "Q": "flow_m3d", "total_cod": "ghost", "tkn": "TKN"}
    with pytest.raises(ValueError, match="ghost"):
        _influent_from_column_map(_LAB_CSV, asm1, bad, None, None, "<t>")


def test_column_map_direct_species_only(asm1):
    # a file already holding ASM states under renamed headers, no fractionation
    csv = "time,Qm3d,readilyCOD,amm\n0,1000,55,30\n1,1200,60,32\n"
    cm = {"t": "time", "Q": "Qm3d", "SS": "readilyCOD", "SNH": "amm"}
    inf = _influent_from_column_map(csv, asm1, cm, None, None, "<t>")
    si = asm1.species_index
    assert float(inf.C[0, si["SS"]]) == 55.0
    assert float(inf.C[1, si["SNH"]]) == 32.0
    assert float(inf.C[0, si["XS"]]) == 0.0        # unmapped -> zero
