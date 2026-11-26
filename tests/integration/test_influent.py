"""Tests for influent CSV / text parsing."""

import jax.numpy as jnp
import pytest

import aquakin
from aquakin.plant.influent import (
    _influent_from_text,
    _looks_like_header,
    _tokenize,
    load_bsm1_influent,
    load_bsm2_influent,
    read_influent_csv,
)


@pytest.fixture
def asm1():
    return aquakin.load_model("asm1")


def _csv_text(model):
    # One header row + two data rows, in the default BSM1 column order.
    from aquakin.plant.influent import _BSM1_COLUMN_ORDER

    header = ",".join(_BSM1_COLUMN_ORDER)
    zeros = dict.fromkeys(_BSM1_COLUMN_ORDER, 0.0)
    rows = []
    for i, tval in enumerate((0.0, 0.25)):
        vals = dict(zeros, t=tval, Q=18000.0 + i, SNH=30.0, SS=70.0)
        rows.append(",".join(f"{vals[c]:g}" for c in _BSM1_COLUMN_ORDER))
    return header + "\n" + "\n".join(rows) + "\n"


def test_influent_constant_is_zero_based(asm1):
    """``model.influent`` builds a zero-based constant feed: only the named
    species are present, everything else is absent (not at its YAML default)."""
    from aquakin.plant.influent import InfluentSeries

    inf = asm1.influent({"SS": 60.0, "SNH": 25.0}, Q=18446.0)
    assert isinstance(inf, InfluentSeries)
    # Constant over time: same stream at any t.
    s0 = inf.at(0.0)
    s5 = inf.at(5.0e3)
    assert float(s0.Q) == pytest.approx(18446.0)
    assert float(s5.Q) == pytest.approx(18446.0)
    assert float(s0.C[asm1.species_index["SS"]]) == pytest.approx(60.0)
    assert float(s0.C[asm1.species_index["SNH"]]) == pytest.approx(25.0)
    # Unlisted species are zero (would be nonzero if defaults leaked in).
    assert float(s0.C[asm1.species_index["XI"]]) == 0.0
    assert float(s0.C[asm1.species_index["SALK"]]) == 0.0


def test_influent_carries_temperature(asm1):
    inf = asm1.influent({"SS": 400.0}, Q=2.0, T=288.15)
    s = inf.at(1.0)
    assert s.scalars.get("T") is not None
    assert float(s.scalars["T"]) == pytest.approx(288.15)
    # No T given -> temperature-agnostic.
    assert asm1.influent(SS=400.0, Q=2.0).at(1.0).scalars.get("T") is None


def test_influent_series_constant_classmethod(asm1):
    """``InfluentSeries.constant`` is the same builder as ``model.influent``."""
    from aquakin.plant.influent import InfluentSeries

    a = asm1.influent({"SS": 60.0}, Q=100.0)
    b = InfluentSeries.constant(asm1, {"SS": 60.0}, Q=100.0)
    assert jnp.allclose(a.C, b.C)
    assert jnp.allclose(a.Q, b.Q)


def test_influent_base_defaults_keeps_reference_values(asm1):
    """base='defaults' starts from the YAML reference composition."""
    inf = asm1.influent({"SS": 60.0}, Q=100.0, base="defaults")
    s = inf.at(0.0)
    # XI keeps its (nonzero) YAML default under base='defaults'.
    assert float(s.C[asm1.species_index["XI"]]) == pytest.approx(
        float(asm1.default_concentrations()[asm1.species_index["XI"]])
    )


def test_tokenize_matches_the_positional_and_headered_parsers():
    """The single tokenizer reproduces both call sites: comma-count-aware when a
    field count is expected, comma-if-present otherwise, explicit delimiter wins."""
    # No delimiter, count expected: comma-split iff it yields the count.
    assert _tokenize("1, 2 ,3", None, expected_n=3) == ["1", "2", "3"]
    assert _tokenize("1 2 3", None, expected_n=3) == ["1", "2", "3"]
    # Comma count mismatch -> fall back to whitespace.
    assert _tokenize("1,2 3", None, expected_n=3) == ["1,2", "3"]
    # No count required: comma-split only when a comma is present (the old
    # ``_split_row`` contract used by the headered-table parser).
    assert _tokenize("1, 2, 3", None) == ["1", "2", "3"]
    assert _tokenize("1 2 3", None) == ["1", "2", "3"]
    # Explicit delimiter is used verbatim (count ignored).
    assert _tokenize("a;b;c", ";", expected_n=99) == ["a", "b", "c"]


def test_looks_like_header_distinguishes_names_from_numbers():
    assert _looks_like_header("t, SS, Q") is True
    assert _looks_like_header("0.0 1.0 2.0") is False
    # Scientific notation contains a letter but parses numerically -> data.
    assert _looks_like_header("1e5 2.0 3") is False
    # Pure symbols, no letters -> not a header.
    assert _looks_like_header("0, 0, 0") is False


def test_influent_from_text_parses_without_a_file(asm1):
    series = _influent_from_text(_csv_text(asm1), asm1)
    assert series.t.shape == (2,)
    assert series.C.shape == (2, asm1.n_species)
    assert float(series.Q[0]) == pytest.approx(18000.0)
    assert float(series.C[0, asm1.species_index["SNH"]]) == pytest.approx(30.0)


def test_read_influent_csv_matches_text(asm1, tmp_path):
    text = _csv_text(asm1)
    p = tmp_path / "inf.csv"
    p.write_text(text)
    from_file = read_influent_csv(p, asm1)
    from_text = _influent_from_text(text, asm1)
    assert jnp.allclose(from_file.C, from_text.C)
    assert jnp.allclose(from_file.Q, from_text.Q)


def test_read_influent_csv_missing_file(asm1, tmp_path):
    with pytest.raises(FileNotFoundError, match="Influent file not found"):
        read_influent_csv(tmp_path / "nope.csv", asm1)


def test_influent_from_text_captures_T_column(asm1):
    """A 'T' column is captured into InfluentSeries.T (in the file's units);
    its absence leaves T = None."""
    from aquakin.plant.influent import _BSM1_COLUMN_ORDER

    order = _BSM1_COLUMN_ORDER + ["T"]
    header = ",".join(order)
    zeros = dict.fromkeys(order, 0.0)
    rows = []
    for tval, temp in ((0.0, 12.0), (0.25, 16.0)):
        vals = dict(zeros, t=tval, Q=18000.0, T=temp)
        rows.append(",".join(f"{vals[c]:g}" for c in order))
    text = header + "\n" + "\n".join(rows) + "\n"
    series = _influent_from_text(text, asm1, column_order=order)
    assert series.T is not None
    assert float(series.T[0]) == pytest.approx(12.0)
    assert float(series.T[1]) == pytest.approx(16.0)
    # No T column -> T is None (default, back-compatible).
    assert _influent_from_text(_csv_text(asm1), asm1).T is None


def test_load_bsm2_influent_carries_temperature_kelvin(asm1):
    """The BSM2 influent files carry a temperature column; load_bsm2_influent
    returns it in Kelvin (the degC file values + 273.15), seasonally varying."""
    for profile in ("dry", "rain", "storm"):
        inf = load_bsm2_influent(profile, asm1)
        assert inf.T is not None
        assert inf.T.shape == inf.t.shape
        # 11.5-18.5 degC window -> ~284.6-291.7 K.
        assert 283.0 < float(inf.T.min()) < float(inf.T.max()) < 293.0
        # at(t) interpolates the temperature onto the returned stream.
        s = inf.at(jnp.asarray(7.0))
        assert s.scalars.get("T") is not None and 285.0 < float(s.scalars["T"]) < 291.0


def test_load_bsm1_influent_has_no_temperature(asm1):
    """BSM1 files have no temperature column, so T stays None (back-compatible)."""
    assert load_bsm1_influent("dry", asm1).T is None


@pytest.mark.parametrize("profile", ["dry", "rain", "storm"])
def test_load_bsm1_influent_no_tempfile(asm1, profile, monkeypatch):
    # The package data must be parsed directly from text; loading must not
    # create a temporary file (the old disk round-trip).
    import tempfile

    def _boom(*a, **k):
        raise AssertionError("load_bsm1_influent must not use a tempfile")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", _boom)
    series = load_bsm1_influent(profile, asm1)
    assert series.t.shape[0] > 1
    assert series.C.shape[1] == asm1.n_species


# --------------------------------------------------------------------------- #
# InfluentSeries.__post_init__ shape validation
# --------------------------------------------------------------------------- #


def test_influent_series_rejects_2d_t(asm1):
    """A 2-D ``t`` array is rejected."""
    from aquakin.plant.influent import InfluentSeries

    t = jnp.zeros((2, 1))
    Q = jnp.zeros(2)
    C = jnp.zeros((2, asm1.n_species))
    with pytest.raises(ValueError, match="t must be 1-D"):
        InfluentSeries(t=t, Q=Q, C=C, model=asm1)


def test_influent_series_rejects_mismatched_Q_shape(asm1):
    """``Q`` whose shape differs from ``t`` is rejected."""
    from aquakin.plant.influent import InfluentSeries

    t = jnp.zeros(2)
    Q = jnp.zeros(3)
    C = jnp.zeros((2, asm1.n_species))
    with pytest.raises(ValueError, match=r"Q shape .* does not match t shape"):
        InfluentSeries(t=t, Q=Q, C=C, model=asm1)


def test_influent_series_rejects_wrong_C_rows(asm1):
    """A ``C`` whose row count differs from ``t`` (or is not 2-D) is rejected."""
    from aquakin.plant.influent import InfluentSeries

    t = jnp.zeros(2)
    Q = jnp.zeros(2)
    C = jnp.zeros((3, asm1.n_species))  # 3 rows vs 2 times
    with pytest.raises(ValueError, match=r"C shape .* expected \(2, n_species\)"):
        InfluentSeries(t=t, Q=Q, C=C, model=asm1)


def test_influent_series_rejects_wrong_species_columns(asm1):
    """A ``C`` with the wrong number of species columns is rejected."""
    from aquakin.plant.influent import InfluentSeries

    t = jnp.zeros(2)
    Q = jnp.zeros(2)
    C = jnp.zeros((2, asm1.n_species + 1))  # one column too many
    with pytest.raises(ValueError, match="species columns but model has"):
        InfluentSeries(t=t, Q=Q, C=C, model=asm1)


def test_influent_series_rejects_mismatched_T_shape(asm1):
    """A ``T`` whose shape differs from ``t`` is rejected."""
    from aquakin.plant.influent import InfluentSeries

    t = jnp.zeros(2)
    Q = jnp.zeros(2)
    C = jnp.zeros((2, asm1.n_species))
    T = jnp.zeros(3)
    with pytest.raises(ValueError, match=r"T shape .* does not match t shape"):
        InfluentSeries(t=t, Q=Q, C=C, model=asm1, T=T)


# --------------------------------------------------------------------------- #
# _influent_from_text positional-parser validation
# --------------------------------------------------------------------------- #


def test_influent_from_text_rejects_wrong_field_count(asm1):
    """A data row with the wrong number of fields is rejected."""
    from aquakin.plant.influent import _BSM1_COLUMN_ORDER

    header = ",".join(_BSM1_COLUMN_ORDER)
    bad_row = ",".join(["0.0"] * (len(_BSM1_COLUMN_ORDER) - 1))  # one field short
    text = header + "\n" + bad_row + "\n"
    with pytest.raises(ValueError, match="fields but"):
        _influent_from_text(text, asm1)


def test_influent_from_text_rejects_non_numeric_field(asm1):
    """A non-numeric field in a data row is rejected. The first row must parse
    as data (a non-numeric first row is swallowed as a column-name header), so a
    good row precedes the bad one."""
    from aquakin.plant.influent import _BSM1_COLUMN_ORDER

    header = ",".join(_BSM1_COLUMN_ORDER)
    good = ",".join(["0.0"] * len(_BSM1_COLUMN_ORDER))
    bad_vals = ["0.0"] * len(_BSM1_COLUMN_ORDER)
    bad_vals[1] = "not_a_number"
    text = header + "\n" + good + "\n" + ",".join(bad_vals) + "\n"
    with pytest.raises(ValueError, match="non-numeric field"):
        _influent_from_text(text, asm1)


def test_influent_from_text_rejects_no_data_rows(asm1):
    """Text with no data rows (only comments) is rejected."""
    text = "# just a comment\n# and another\n"
    with pytest.raises(ValueError, match="contained no data rows"):
        _influent_from_text(text, asm1, source="<empty>")


def test_influent_from_text_requires_Q_column(asm1):
    """A ``column_order`` without a 'Q' column is rejected."""
    order = [c for c in ["t", "SS", "SNH"] if c != "Q"]  # no Q
    header = ",".join(order)
    text = header + "\n" + ",".join(["0.0"] * len(order)) + "\n"
    with pytest.raises(ValueError, match="column_order must contain 'Q'"):
        _influent_from_text(text, asm1, column_order=order)


def test_influent_from_text_requires_all_species_columns(asm1):
    """A ``column_order`` missing a model species column is rejected."""
    order = ["t", "Q", "SS"]  # missing most ASM1 species
    header = ",".join(order)
    text = header + "\n" + ",".join(["0.0"] * len(order)) + "\n"
    with pytest.raises(ValueError, match="missing species column"):
        _influent_from_text(text, asm1, column_order=order)


# --------------------------------------------------------------------------- #
# _influent_from_column_map (headered-table) validation
# --------------------------------------------------------------------------- #


def test_column_map_rejects_non_numeric_field(asm1):
    """A non-numeric data field in a column_map-parsed table is rejected."""
    text = "time,flow\n0.0,oops\n"
    column_map = {"t": "time", "Q": "flow"}
    with pytest.raises(ValueError, match="non-numeric field"):
        _influent_from_text(text, asm1, column_map=column_map)


def test_column_map_requires_header_and_data(asm1):
    """A column_map table with a header but no data rows is rejected."""
    text = "time,flow\n"  # header only, no data
    column_map = {"t": "time", "Q": "flow"}
    with pytest.raises(ValueError, match="header row and at least one data row"):
        _influent_from_text(text, asm1, column_map=column_map)


def test_column_map_requires_t_and_Q_roles(asm1):
    """A column_map missing the required 't'/'Q' roles is rejected."""
    text = "time,flow\n0.0,1.0\n"
    column_map = {"t": "time"}  # no 'Q' role
    with pytest.raises(ValueError, match="column_map must map the 'Q' role"):
        _influent_from_text(text, asm1, column_map=column_map)


def test_column_map_rejects_role_missing_from_header(asm1):
    """A column_map role pointing at a header name not in the file is rejected."""
    text = "time,flow\n0.0,1.0\n"
    column_map = {"t": "time", "Q": "missing_col"}
    with pytest.raises(ValueError, match="is not in the file header"):
        _influent_from_text(text, asm1, column_map=column_map)


# --------------------------------------------------------------------------- #
# load_bsm1_influent / load_bsm2_influent profile + missing-file validation
# --------------------------------------------------------------------------- #


def test_load_bsm1_influent_rejects_bad_profile(asm1):
    with pytest.raises(ValueError, match="profile must be 'dry', 'rain', or 'storm'"):
        load_bsm1_influent("sunny", asm1)


def test_load_bsm2_influent_rejects_bad_profile(asm1):
    with pytest.raises(ValueError, match="profile must be 'dry', 'rain', or 'storm'"):
        load_bsm2_influent("sunny", asm1)


def test_load_bsm1_influent_missing_file(asm1, monkeypatch):
    """A valid profile whose package-data file is absent raises FileNotFoundError."""
    import aquakin.plant.influent as influent_mod

    class _FakeResource:
        def __truediv__(self, other):
            return self

        def is_file(self):
            return False

    monkeypatch.setattr(influent_mod, "files", lambda pkg: _FakeResource())
    with pytest.raises(FileNotFoundError, match="BSM1 influent file"):
        load_bsm1_influent("dry", asm1)


def test_load_bsm2_influent_missing_file(asm1, monkeypatch):
    """A valid profile whose package-data file is absent raises FileNotFoundError."""
    import aquakin.plant.influent as influent_mod

    class _FakeResource:
        def __truediv__(self, other):
            return self

        def is_file(self):
            return False

    monkeypatch.setattr(influent_mod, "files", lambda pkg: _FakeResource())
    with pytest.raises(FileNotFoundError, match="BSM2 influent file"):
        load_bsm2_influent("dry", asm1)


# ----- a record run past its ends warns ------------------------------------


def _ramp_plant(simple_model, t_end=100.0, t_start=0.0):
    """One CSTR on the decay model fed by a record whose flow ramps from 5 to 15
    over ``[t_start, t_end]`` (plant clock: seconds)."""
    from aquakin.plant import CSTRUnit, InfluentSeries, Plant

    plant = Plant("ramp", time_unit="s")
    plant.add_unit(
        CSTRUnit(name="tank", model=simple_model, volume=100.0, input_port_names=["inlet"],
                 conditions={"T": 293.15})
    )
    record = InfluentSeries(
        t=jnp.asarray([t_start, t_end]),
        Q=jnp.asarray([5.0, 15.0]),
        C=jnp.asarray([[1.0, 0.0], [1.0, 0.0]]),
        model=simple_model,
    )
    plant.add_influent("ramp_feed", record, to="tank.inlet")
    return plant


def _range_warnings(record):
    from aquakin.plant import InfluentRangeWarning

    return [w for w in record if issubclass(w.category, InfluentRangeWarning)]


def test_solve_past_the_last_sample_warns_once_naming_the_influent(simple_model):
    """A solve that ends after a time-varying record's last sample warns once,
    naming the influent, the record's range and the requested span; following
    its advice (a record covering the span) silences it."""
    import warnings

    from aquakin.plant import InfluentRangeWarning

    plant = _ramp_plant(simple_model)
    with pytest.warns(InfluentRangeWarning) as record:
        plant.solve(t_span=(0.0, 150.0))
    (w,) = _range_warnings(record)
    message = str(w.message)
    assert "'ramp_feed'" in message
    assert "t = 0 to 100" in message
    assert "t = 0 to 150" in message
    assert "last sample" in message

    covered = _ramp_plant(simple_model, t_end=150.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", InfluentRangeWarning)
        sol = covered.solve(t_span=(0.0, 150.0))
    assert float(sol.t[-1]) == pytest.approx(150.0, rel=1e-12)


def test_solve_before_the_first_sample_warns(simple_model):
    """A solve starting before a record's first sample warns about the start."""
    from aquakin.plant import InfluentRangeWarning

    plant = _ramp_plant(simple_model, t_start=10.0, t_end=100.0)
    with pytest.warns(InfluentRangeWarning, match="before its first sample") as record:
        plant.solve(t_span=(0.0, 50.0))
    (w,) = _range_warnings(record)
    assert "t = 10 to 100" in str(w.message)


def test_solve_within_the_record_does_not_warn(simple_model):
    """The absence of the warning is the property under test: a span inside the
    record, and a constant feed (two equal samples, or ``InfluentSeries.constant``)
    run far past its samples, do not warn."""
    import warnings

    from aquakin.plant import CSTRUnit, InfluentRangeWarning, InfluentSeries, Plant

    plant = _ramp_plant(simple_model)
    flat = Plant("flat", time_unit="s")
    flat.add_unit(CSTRUnit(name="tank", model=simple_model, volume=100.0,
                           input_port_names=["inlet"], conditions={"T": 293.15}))
    flat.add_influent(
        "flat_feed",
        InfluentSeries(t=jnp.asarray([0.0, 10.0]), Q=jnp.asarray([10.0, 10.0]),
                       C=jnp.asarray([[1.0, 0.0], [1.0, 0.0]]), model=simple_model),
        to="tank.inlet",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", InfluentRangeWarning)
        plant.solve(t_span=(0.0, 100.0))
        flat.solve(t_span=(0.0, 500.0))


def test_solve_sensitivity_past_the_last_sample_warns(simple_model):
    """The forward-sensitivity solve checks the span too."""
    from aquakin.plant import InfluentRangeWarning

    plant = _ramp_plant(simple_model)
    with pytest.warns(InfluentRangeWarning, match="'ramp_feed'"):
        plant.solve_sensitivity(
            plant.default_parameters(), ["simple_decay.A_to_B.k"], t_span=(0.0, 150.0)
        )


def test_traced_records_warn_only_when_their_range_is_known(simple_model):
    """Under a trace (a record built inside ``jax.jit``) a record whose values are
    traced cannot be shown constant, so a span past its concrete range warns;
    a record whose sample times are traced has no range to compare, so it does
    not warn."""
    import warnings

    import jax

    from aquakin.plant import InfluentRangeWarning
    from aquakin.plant.influent import InfluentSeries, warn_if_outside_records

    C = jnp.asarray([[1.0, 0.0], [1.0, 0.0]])
    t = jnp.asarray([0.0, 100.0])  # built outside the trace, so concrete inside it

    @jax.jit
    def traced_values(Q):
        record = InfluentSeries(t=t, Q=Q, C=C, model=simple_model)
        warn_if_outside_records({"feed": record}, 0.0, 150.0, "s")
        return Q

    @jax.jit
    def traced_times(times):
        record = InfluentSeries(t=times, Q=jnp.asarray([5.0, 15.0]), C=C, model=simple_model)
        warn_if_outside_records({"feed": record}, 0.0, 150.0, "s")
        return times

    with pytest.warns(InfluentRangeWarning, match="t = 0 to 100 s"):
        traced_values(jnp.asarray([10.0, 10.0]))
    with warnings.catch_warnings():
        warnings.simplefilter("error", InfluentRangeWarning)
        traced_times(jnp.asarray([0.0, 100.0]))
