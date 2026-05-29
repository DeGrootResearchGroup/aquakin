"""End-to-end Lagrangian-coupling demo.

Synthesises a small ensemble of particle tracks (without needing a real
OpenFOAM run), writes them through the CSV track format, reads them back,
integrates the expanded ozone/bromate network along each, and prints a
summary of bromate yields plus the RTD Morrill index of the synthetic
tracer.
"""

from pathlib import Path

import jax.numpy as jnp

import aquakin
from aquakin.transport.openfoam import read_tracks_csv, write_tracks_csv
from aquakin.utils import rtd


def _synthesise_tracks(n_particles: int, t_end: float, n_samples: int):
    """Build n_particles tracks with different OH-scavenging plateaus."""
    t = jnp.linspace(0.0, t_end, n_samples)
    tracks: dict[int, aquakin.Track] = {}
    # Scavenging varies from low (clean water, 1e3 s-1) to high (DOC-rich, 1e6).
    scav_levels = jnp.logspace(3, 6, n_particles)
    for pid in range(n_particles):
        tracks[pid] = aquakin.Track(
            t=t,
            fields={
                "pH": jnp.full_like(t, 7.5),
                "T": jnp.full_like(t, 293.15),
                "OH_scavenging": jnp.full_like(t, float(scav_levels[pid])),
            },
        )
    return tracks


def _synthetic_tracer(t_end: float, n_samples: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Synthetic outlet tracer response (mix of CSTR + small bypass)."""
    t = jnp.linspace(0.0, t_end, n_samples)
    tau = 200.0
    C = 0.9 * jnp.exp(-t / tau) + 0.1 * jnp.exp(-((t - 30.0) / 5.0) ** 2)
    return t, C


def main() -> None:
    network = aquakin.load_network("ozone_bromate")
    atol = jnp.full((network.n_species,), 1e-12)
    atol = atol.at[network.species_index["OH"]].set(1e-20)

    t_end = 600.0
    n_samples = 61
    n_particles = 5

    tracks = _synthesise_tracks(n_particles, t_end, n_samples)

    csv_path = Path("particles.csv")
    write_tracks_csv(csv_path, tracks)
    print(f"Wrote {csv_path} ({n_particles} particles, {n_samples} samples each).")
    loaded = read_tracks_csv(csv_path)
    print(f"Loaded {len(loaded)} particles back from CSV.")

    def C0_fn(_pid: int) -> jnp.ndarray:
        C0 = network.default_concentrations()
        C0 = C0.at[network.species_index["O3"]].set(1.0e-4)
        C0 = C0.at[network.species_index["Br-"]].set(1.0e-5)
        return C0

    solutions = aquakin.integrate_ensemble(
        network,
        loaded,
        C0_fn=C0_fn,
        params=network.default_parameters(),
        atol=atol,
    )

    print()
    print(f"{'pid':>4}  {'OH_scav [1/s]':>14}  {'BrO3- [M]':>14}  {'OH peak [M]':>14}")
    for pid, sol in solutions.items():
        scav = float(loaded[pid].fields["OH_scavenging"][0])
        bro3 = float(sol.C_named("BrO3-")[-1])
        oh_peak = float(jnp.max(sol.C_named("OH")))
        print(f"{pid:>4d}  {scav:>14.3e}  {bro3:>14.4e}  {oh_peak:>14.4e}")

    print()
    t_tracer, C_tracer = _synthetic_tracer(t_end, n_samples)
    morrill = float(rtd.morrill_index(t_tracer, C_tracer))
    mean = float(rtd.mean_residence_time(t_tracer, C_tracer))
    print(f"Synthetic tracer: mean residence time = {mean:.1f} s, Morrill = {morrill:.2f}")


if __name__ == "__main__":
    main()
