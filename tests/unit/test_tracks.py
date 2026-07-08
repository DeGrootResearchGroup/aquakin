"""Unit tests for the CSV track reader/writer."""

import jax.numpy as jnp
import pytest

import aquakin
from aquakin.transport.openfoam import read_tracks_csv, write_tracks_csv


def _make_tracks():
    return {
        0: aquakin.Track(
            t=jnp.asarray([0.0, 1.0, 2.0]),
            fields={
                "pH": jnp.asarray([7.5, 7.6, 7.7]),
                "T": jnp.asarray([293.15, 293.2, 293.3]),
            },
        ),
        2: aquakin.Track(
            t=jnp.asarray([0.0, 1.0, 2.0]),
            fields={
                "pH": jnp.asarray([8.0, 8.0, 8.0]),
                "T": jnp.asarray([295.0, 295.5, 296.0]),
            },
        ),
    }


def test_round_trip(tmp_path):
    tracks = _make_tracks()
    csv_path = tmp_path / "tracks.csv"
    write_tracks_csv(csv_path, tracks)
    loaded = read_tracks_csv(csv_path)
    assert set(loaded.keys()) == set(tracks.keys())
    for pid, original in tracks.items():
        rt = loaded[pid]
        assert jnp.allclose(rt.t, original.t)
        for name in original.fields:
            assert jnp.allclose(rt.fields[name], original.fields[name])


def test_missing_particle_id_column_rejected(tmp_path):
    p = tmp_path / "tracks.csv"
    p.write_text("t,pH\n0.0,7.5\n1.0,7.5\n")
    with pytest.raises(ValueError):
        read_tracks_csv(p)


def test_non_ascending_t_rejected(tmp_path):
    p = tmp_path / "tracks.csv"
    p.write_text("particle_id,t,pH\n0,0.0,7.5\n0,2.0,7.5\n0,1.0,7.5\n")
    with pytest.raises(ValueError):
        read_tracks_csv(p)


def test_non_integer_particle_id_rejected(tmp_path):
    p = tmp_path / "tracks.csv"
    p.write_text("particle_id,t,pH\nfoo,0.0,7.5\n")
    with pytest.raises(ValueError):
        read_tracks_csv(p)


def test_row_length_mismatch_rejected(tmp_path):
    p = tmp_path / "tracks.csv"
    p.write_text("particle_id,t,pH,T\n0,0.0,7.5\n")
    with pytest.raises(ValueError):
        read_tracks_csv(p)


def test_inconsistent_field_names_rejected_on_write(tmp_path):
    tracks = {
        0: aquakin.Track(t=jnp.asarray([0.0, 1.0]), fields={"pH": jnp.asarray([7.0, 7.0])}),
        1: aquakin.Track(t=jnp.asarray([0.0, 1.0]), fields={"T": jnp.asarray([293.15, 293.15])}),
    }
    with pytest.raises(ValueError):
        write_tracks_csv(tmp_path / "out.csv", tracks)


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        read_tracks_csv("/no/such/file.csv")


def test_empty_file_rejected(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("")
    with pytest.raises(ValueError, match="Empty track file"):
        read_tracks_csv(p)


def test_non_numeric_field_value_rejected(tmp_path):
    p = tmp_path / "tracks.csv"
    p.write_text("particle_id,t,pH\n0,0.0,notanumber\n")
    with pytest.raises(ValueError, match="non-numeric value"):
        read_tracks_csv(p)


def test_non_finite_field_value_rejected(tmp_path):
    p = tmp_path / "tracks.csv"
    p.write_text("particle_id,t,pH\n0,0.0,inf\n")
    with pytest.raises(ValueError, match="non-finite value"):
        read_tracks_csv(p)


def test_duplicate_t_rejected(tmp_path):
    p = tmp_path / "tracks.csv"
    p.write_text("particle_id,t,pH\n0,0.0,7.5\n0,0.0,7.6\n")
    with pytest.raises(ValueError, match="duplicate t values"):
        read_tracks_csv(p)


def test_empty_tracks_mapping_rejected_on_write(tmp_path):
    with pytest.raises(ValueError, match="empty tracks mapping"):
        write_tracks_csv(tmp_path / "out.csv", {})


def test_bridge_field_shape_mismatch_rejected():
    from aquakin.transport.openfoam.bridge import from_cell_fields

    with pytest.raises(ValueError, match="expected"):
        from_cell_fields({"pH": jnp.asarray([7.0, 7.5])}, n_cells=3)


def test_bridge_builds_spatial_conditions():
    from aquakin.transport.openfoam.bridge import from_cell_fields

    cond = from_cell_fields({"pH": jnp.asarray([7.0, 7.5, 8.0])}, n_cells=3)
    assert cond.n_locations == 3
    assert jnp.allclose(cond.fields["pH"], jnp.asarray([7.0, 7.5, 8.0]))
