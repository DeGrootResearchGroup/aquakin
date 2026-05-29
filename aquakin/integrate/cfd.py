"""CFD-coupled batch chemistry reactor.

:class:`CFDReactor` is the Python entry point for runtime coupling with a
CFD solver (e.g. OpenFOAM Option C in CLAUDE.md). It vectorises the
single-cell stiff-chemistry sub-problem over every CFD cell using
``jax.vmap``, then returns post-reaction concentrations as a NumPy array.

The intended usage from a C++ ``fvOptions``-style plugin is::

    reactor = aquakin.CFDReactor(network)
    # ... per timestep ...
    C_new = reactor.step(
        C,            # (n_cells, n_species)   float64 NumPy
        conditions,   # {name: (n_cells,)}     float64 NumPy
        dt,           # scalar float (seconds)
        params,       # (n_params,) NumPy or None
    )

The NumPy boundary keeps the pybind11 binding straightforward — the C++
side hands over contiguous ``double[]`` buffers and receives the same.

Columns of ``C`` follow ``network.species`` order; the C++ side is
responsible for assembling that array from its OpenFOAM volScalarFields in
the right order. The reactor's own
:attr:`CFDReactor.species_field_order` attribute exposes this contract.
"""

from __future__ import annotations

from typing import Mapping, Optional

import diffrax
import jax
import jax.numpy as jnp
import numpy as np

from aquakin.core.network import CompiledNetwork
from aquakin.integrate._common import _coerce_atol, _run_diffeqsolve


class CFDReactor:
    """
    Vectorised batch reactor for CFD operator-splitting.

    Each call to :meth:`step` advances the chemistry from the supplied
    cell-state for a transport sub-step ``dt``. Inside, every cell runs an
    independent stiff Diffrax integration via ``jax.vmap``.

    Parameters
    ----------
    network : CompiledNetwork
        Compiled reaction network.
    rtol : float, optional
        Relative tolerance for the per-cell ODE solver.
    atol : float or jnp.ndarray, optional
        Absolute tolerance. Scalar or shape ``(n_species,)``. See
        :class:`BatchReactor` for the per-species rationale.
    adjoint : diffrax.AbstractAdjoint, optional
        Adjoint strategy. Defaults to
        :class:`diffrax.RecursiveCheckpointAdjoint`.
    on_nan : {"raise", "ignore"}, optional
        Policy when any cell produces a NaN concentration after the
        chemistry step. ``"raise"`` (default) raises ``RuntimeError`` with
        the offending cell indices; the C++ caller is then expected to
        retry with a smaller transport timestep or otherwise recover.

    Attributes
    ----------
    network : CompiledNetwork
    species_field_order : list[str]
        Convenience: the order in which species columns of ``C`` must be
        supplied. Equal to ``network.species``.
    """

    def __init__(
        self,
        network: CompiledNetwork,
        *,
        rtol: float = 1e-6,
        atol=1e-9,
        adjoint: Optional[diffrax.AbstractAdjoint] = None,
        on_nan: str = "raise",
        dtmax: Optional[float] = None,
    ) -> None:
        if on_nan not in ("raise", "ignore"):
            raise ValueError(
                f"on_nan must be 'raise' or 'ignore', got {on_nan!r}"
            )
        self.network = network
        self.rtol = rtol
        self.atol = _coerce_atol(atol, network.n_species)
        self.adjoint = adjoint
        self.on_nan = on_nan
        self.dtmax = dtmax
        # Cache jit-compiled vmapped step keyed on n_cells.
        self._jit_cache: dict[int, callable] = {}

    @property
    def species_field_order(self) -> list[str]:
        """Order in which species columns of ``C`` must be supplied."""
        return list(self.network.species)

    @property
    def condition_field_names(self) -> list[str]:
        """Names of condition fields expected in ``conditions`` dict."""
        return list(self.network.conditions_required)

    def step(
        self,
        C: np.ndarray,
        conditions: Mapping[str, np.ndarray],
        dt: float,
        params: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Advance chemistry by ``dt`` for every cell.

        Parameters
        ----------
        C : np.ndarray
            Cell concentrations, shape ``(n_cells, n_species)``. Columns
            follow ``self.species_field_order``.
        conditions : mapping str -> np.ndarray
            Per-cell condition arrays, each shape ``(n_cells,)``. Must
            include every name in ``self.condition_field_names``.
        dt : float
            Transport sub-step length over which to integrate chemistry.
            Must be positive.
        params : np.ndarray, optional
            Flat parameter vector, shape ``(n_params,)``. Defaults to
            ``network.default_parameters()``.

        Returns
        -------
        np.ndarray
            Post-reaction concentrations, same shape as input ``C``.
        """
        C_np = np.ascontiguousarray(np.asarray(C, dtype=np.float64))
        if C_np.ndim != 2:
            raise ValueError(
                f"C must be 2-D (n_cells, n_species); got shape {C_np.shape}"
            )
        n_cells, n_species_in = C_np.shape
        if n_species_in != self.network.n_species:
            raise ValueError(
                f"C has {n_species_in} species columns but network has "
                f"{self.network.n_species} species ({self.network.species})."
            )
        if n_cells < 1:
            raise ValueError(f"C must have at least 1 row; got {n_cells}")

        missing = set(self.network.conditions_required) - set(conditions)
        if missing:
            raise ValueError(
                f"conditions is missing required field(s): {sorted(missing)}. "
                f"Provided: {sorted(conditions)}"
            )
        cond_jax: dict[str, jnp.ndarray] = {}
        for name in self.network.conditions_required:
            arr = np.asarray(conditions[name], dtype=np.float64)
            if arr.shape != (n_cells,):
                raise ValueError(
                    f"conditions[{name!r}] has shape {arr.shape}, expected "
                    f"({n_cells},)."
                )
            cond_jax[name] = jnp.asarray(arr)

        dt_f = float(dt)
        if not (dt_f > 0):
            raise ValueError(f"dt must be positive; got {dt_f}")

        if params is None:
            params_jax = self.network.default_parameters()
        else:
            params_np = np.asarray(params, dtype=np.float64)
            if params_np.shape != (self.network.n_params,):
                raise ValueError(
                    f"params has shape {params_np.shape}, expected "
                    f"({self.network.n_params},)."
                )
            params_jax = jnp.asarray(params_np)

        inner = self._jit_cache.get(n_cells)
        if inner is None:
            inner = self._build_step()
            self._jit_cache[n_cells] = inner

        C_new = inner(jnp.asarray(C_np), cond_jax, jnp.asarray(dt_f), params_jax)
        C_new_np = np.asarray(C_new)

        if self.on_nan == "raise":
            bad_rows = np.where(np.any(~np.isfinite(C_new_np), axis=1))[0]
            if bad_rows.size:
                raise RuntimeError(
                    f"CFDReactor.step produced non-finite concentrations in "
                    f"{bad_rows.size} cell(s); offending indices (first 10): "
                    f"{bad_rows[:10].tolist()}. Consider reducing dt."
                )
        return C_new_np

    def _build_step(self):
        """Construct the jit-compiled vmapped per-cell step."""
        network = self.network
        rtol = self.rtol
        atol = self.atol
        adjoint = self.adjoint
        dtmax = self.dtmax

        def _per_cell(C_cell, cond_cell, dt, params):
            # cond_cell has scalar values (vmap stripped the cells axis);
            # wrap each in a length-1 array so ConditionNode can index with
            # loc_idx=0.
            cond_arrays = {name: v[None] for name, v in cond_cell.items()}
            # Stoichiometry depends only on params; precompute once per
            # vmapped cell (params is broadcast, so this is hoisted out of
            # the per-cell vmap by the jit compiler anyway).
            stoich = network.compute_stoich(params)

            def rhs(t, C, args):
                return network.dCdt(C, args, cond_arrays, 0, stoich=stoich)

            sol = _run_diffeqsolve(
                rhs,
                t0=0.0,
                t1=dt,
                y0=C_cell,
                args=params,
                saveat=diffrax.SaveAt(t1=True),
                rtol=rtol,
                atol=atol,
                adjoint=adjoint,
                dtmax=dtmax,
            )
            return sol.ys[-1]

        vmapped = jax.vmap(_per_cell, in_axes=(0, 0, None, None))
        return jax.jit(vmapped)
