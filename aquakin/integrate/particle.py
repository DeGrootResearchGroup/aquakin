"""Reactor that integrates kinetics along a single time-varying-condition track.

This is the runtime side of the offline OpenFOAM coupling (Option A in
CLAUDE.md). Each Lagrangian particle, as it traverses the CFD domain,
samples the condition fields (pH, T, scavenging, fluence_rate, ...) along
its path. ``ParticleTrackReactor`` integrates the chemistry along that path
by linearly interpolating each condition field over time inside the ODE
term.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import diffrax
import jax
import jax.numpy as jnp

from aquakin.core.network import CompiledNetwork
from aquakin.integrate._common import (
    _HasNamedSpecies,
    _coerce_atol,
    _interp_fields_to_scalar,
    _run_diffeqsolve,
)


@dataclass
class Track:
    """
    A single particle track: condition fields sampled at successive times.

    Attributes
    ----------
    t : jnp.ndarray
        Ascending sample times, shape ``(n_points,)``.
    fields : dict[str, jnp.ndarray]
        Mapping from condition field name to a 1-D array of length
        ``n_points`` giving that field along the track.
    """

    t: jnp.ndarray
    fields: dict[str, jnp.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        t = jnp.asarray(self.t)
        if t.ndim != 1:
            raise ValueError(f"Track.t must be 1-D, got shape {t.shape}")
        if t.shape[0] < 2:
            raise ValueError("Track must have at least 2 sample points")
        if not bool(jnp.all(jnp.diff(t) > 0)):
            raise ValueError("Track.t must be strictly ascending")
        n = int(t.shape[0])
        normalised: dict[str, jnp.ndarray] = {}
        for name, value in self.fields.items():
            arr = jnp.asarray(value)
            if arr.shape != (n,):
                raise ValueError(
                    f"Track field '{name}' has shape {arr.shape}, expected ({n},)"
                )
            normalised[name] = arr
        self.t = t
        self.fields = normalised

    @property
    def n_points(self) -> int:
        return int(self.t.shape[0])


@dataclass
class TrackSolution(_HasNamedSpecies):
    """Solution returned by :meth:`ParticleTrackReactor.solve`."""

    t: jnp.ndarray
    C: jnp.ndarray
    network: CompiledNetwork


class ParticleTrackReactor:
    """
    Integrate chemistry along a single particle track.

    Parameters
    ----------
    network : CompiledNetwork
    track : Track
        Time series of condition values along the particle path. Must supply
        every field declared in ``network.conditions_required``.
    n_save : int, optional
        Number of times at which to record the solution between ``t[0]`` and
        ``t[-1]``. Defaults to the number of track sample points.
    rtol : float, optional
        Relative tolerance for the ODE solver.
    atol : float or jnp.ndarray, optional
        Absolute tolerance, scalar or shape ``(n_species,)``.
    """

    def __init__(
        self,
        network: CompiledNetwork,
        track: Track,
        *,
        n_save: int | None = None,
        rtol: float = 1e-6,
        atol=1e-9,
        dtmax: float | None = None,
    ) -> None:
        missing = sorted(set(network.conditions_required) - set(track.fields))
        if missing:
            raise ValueError(
                f"Track is missing required condition fields: {missing}. "
                f"Provided: {sorted(track.fields)}"
            )
        self.network = network
        self.track = track
        self.n_save = int(n_save) if n_save is not None else track.n_points
        if self.n_save < 2:
            raise ValueError(f"n_save must be >= 2, got {self.n_save}")
        self.rtol = rtol
        self.atol = _coerce_atol(atol, network.n_species)
        self.dtmax = dtmax
        # Single jitted variant: the track structure is fixed for the
        # reactor's lifetime, so one cache slot is enough.
        self._jitted_solve = None

    def solve(self, C0: jnp.ndarray, params: jnp.ndarray) -> TrackSolution:
        """
        Integrate the network along the track.

        Parameters
        ----------
        C0 : jnp.ndarray
            Inlet concentration vector, shape ``(n_species,)``.
        params : jnp.ndarray
            Flat parameter vector.

        Returns
        -------
        TrackSolution
        """
        C0 = jnp.asarray(C0)
        params = jnp.asarray(params)
        if C0.shape != (self.network.n_species,):
            raise ValueError(
                f"C0 has shape {C0.shape}, expected ({self.network.n_species},)"
            )
        if params.shape != (self.network.n_params,):
            raise ValueError(
                f"params has shape {params.shape}, expected ({self.network.n_params},)"
            )

        if self._jitted_solve is None:
            self._jitted_solve = self._build_jitted_solve()
        ts, ys = self._jitted_solve(C0, params)
        return TrackSolution(t=ts, C=ys, network=self.network)

    def _build_jitted_solve(self):
        """Build a jit-compiled inner solver. Track is closed-over."""
        network = self.network
        t_grid = self.track.t
        fields = self.track.fields
        t0 = float(t_grid[0])
        t1 = float(t_grid[-1])
        t_save = jnp.linspace(t0, t1, self.n_save)
        rtol = self.rtol
        atol = self.atol
        dtmax = self.dtmax

        @jax.jit
        def _solve(C0, params):
            stoich = network.compute_stoich(params)

            def rhs(t, C, args):
                params_ = args
                cond = _interp_fields_to_scalar(t, t_grid, fields)
                return network.dCdt(C, params_, cond, 0, stoich=stoich)

            sol = _run_diffeqsolve(
                rhs,
                t0=t0,
                t1=t1,
                y0=C0,
                args=params,
                saveat=diffrax.SaveAt(ts=t_save),
                rtol=rtol,
                atol=atol,
                dtmax=dtmax,
            )
            return sol.ts, sol.ys

        return _solve


def integrate_ensemble(
    network: CompiledNetwork,
    tracks: Mapping[int, Track],
    C0_fn,
    params: jnp.ndarray,
    *,
    rtol: float = 1e-6,
    atol=1e-9,
    n_save: int | None = None,
) -> dict[int, TrackSolution]:
    """
    Integrate the network along an ensemble of particle tracks.

    Parameters
    ----------
    network : CompiledNetwork
    tracks : mapping int -> Track
        ``particle_id -> Track``.
    C0_fn : callable
        Maps ``particle_id`` to its inlet concentration vector. Often
        ``lambda pid: network.default_concentrations()``.
    params : jnp.ndarray
        Flat parameter vector shared across all particles.
    rtol, atol, n_save : passed through to each :class:`ParticleTrackReactor`.

    Returns
    -------
    dict[int, TrackSolution]
        One solution per particle, keyed by id.
    """
    results: dict[int, TrackSolution] = {}
    for pid, track in tracks.items():
        reactor = ParticleTrackReactor(
            network, track, n_save=n_save, rtol=rtol, atol=atol
        )
        results[pid] = reactor.solve(C0_fn(pid), params)
    return results
