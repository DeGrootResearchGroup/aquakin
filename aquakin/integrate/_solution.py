"""The solution-object mixins (species by name, plotting, tabular export) and
the optional pandas / matplotlib imports they need.

Not part of the public API; ``aquakin.integrate._common`` re-exports every name.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from aquakin.core.hints import did_you_mean
from aquakin.core.model import CompiledModel

# --- Tabular export helpers (optional pandas) --------------------------------


def require_pandas():
    """Import and return pandas, with a helpful message if it is missing.

    pandas is an optional dependency, used only by the ``to_dataframe()`` /
    ``to_csv()`` result exporters.
    """
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover - exercised only without pandas
        raise ImportError(
            "to_dataframe() / to_csv() require pandas, an optional dependency. "
            "Install it with `pip install pandas` or `pip install "
            "aquakin[dataframe]`."
        ) from e
    return pd


def require_matplotlib():
    """Import and return ``matplotlib.pyplot``, with a helpful message if missing.

    matplotlib is an optional dependency, used only by the ``plot()`` result
    helpers.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover - exercised only without mpl
        raise ImportError(
            "plot() requires matplotlib, an optional dependency. Install it with "
            "`pip install matplotlib` or `pip install aquakin[plot]`."
        ) from e
    return plt


def build_dataframe(
    index,
    columns,
    *,
    index_name=None,
    units=None,
    units_in_columns=False,
    extra=None,
):
    """Assemble a pandas ``DataFrame`` from an index and named value columns.

    Shared by every result exporter so the column/units conventions stay
    consistent.

    Parameters
    ----------
    index : array-like or pandas.Index
        The row index. A plain array is wrapped in a ``pd.Index`` named
        ``index_name``; a pre-built ``Index``/``MultiIndex`` is used as-is.
    columns : list of (str, array)
        ``(species_name, 1-D values)`` pairs, in display order.
    index_name : str, optional
        Name for the index when ``index`` is a plain array.
    units : dict, optional
        ``{species_name: unit_string}``; stored in ``df.attrs["units"]`` and,
        when ``units_in_columns`` is set, appended to the column labels.
    units_in_columns : bool, optional
        Append ``" [unit]"`` to each column label.
    extra : list of (str, array), optional
        Additional non-species columns to place before the value columns (e.g.
        a flow ``Q`` or a ``depth`` column). Never relabelled with units.

    Returns
    -------
    pandas.DataFrame
    """
    pd = require_pandas()
    units = units or {}
    if isinstance(index, (pd.Index, pd.MultiIndex)):
        idx = index
    else:
        idx = pd.Index(np.asarray(index), name=index_name)
    data = {}
    for name, arr in extra or []:
        data[name] = np.asarray(arr)
    for name, arr in columns:
        unit = units.get(name, "")
        label = f"{name} [{unit}]" if (units_in_columns and unit) else name
        data[label] = np.asarray(arr)
    df = pd.DataFrame(data, index=idx)
    df.attrs["units"] = dict(units)
    return df


class _HasNamedSpecies:
    """Mixin: the pure species-by-name accessor over a ``.C`` array and ``.model``.

    Solution dataclasses inherit from this to share the species-by-name accessor
    (``C_named`` / ``C_named_many`` / ``final_named`` / ``final`` / ``units_named``
    / ``time_unit``) plus the independent-axis description (``_table_index`` /
    ``_independent_axis_label``) that the presentation mixins build on. The two
    presentation concerns are **separate mixins** -- :class:`PlottableSolutionMixin`
    (matplotlib) and :class:`ExportableSolutionMixin` (pandas / CSV) -- composed
    alongside this one, so a change to the plotting API or the CSV schema (and
    their optional dependencies) never touches the core numeric accessor every
    reactor depends on.
    """

    C: jnp.ndarray  # set by the dataclass subclass
    model: CompiledModel  # set by the dataclass subclass

    def C_named(self, species: str) -> jnp.ndarray:
        """Return the trajectory of a single species by name."""
        if species not in self.model.species_index:
            suffix = did_you_mean(species, self.model.species)
            raise KeyError(f"Unknown species '{species}'. Available: {self.model.species}.{suffix}")
        return self.C[:, self.model.species_index[species]]

    def C_named_many(self, species) -> dict[str, jnp.ndarray]:
        """Trajectories of several species by name, as ``{name: array}``.

        The multi-species companion to :meth:`C_named` -- read a handful of
        species in one call (``sol.C_named_many(["SNH", "SNO", "SO"])``) instead
        of a separate slice each. Each value has the same shape ``C_named``
        returns. An unknown name raises the same hinted ``KeyError``.
        """
        return {sp: self.C_named(sp) for sp in species}

    def final_named(self, species=None) -> dict[str, float]:
        """Values at the **last** recorded point, as ``{name: float}``.

        The reporting shortcut for a steady-state / end-of-run value: instead of
        ``float(sol.C_named("SNH")[-1])`` per species, ``sol.final_named(["SNH",
        "SNO"])`` returns them in one dict. With ``species=None`` (default) every
        model species is returned. Values are plain Python floats (this is a
        post-processing read on an already-solved solution -- use
        ``C_named(sp)[-1]`` if you need a differentiable last value).
        """
        names = list(self.model.species) if species is None else list(species)
        return {sp: float(self.C_named(sp)[-1]) for sp in names}

    @property
    def final(self) -> dict[str, float]:
        """Every species' value at the last recorded point (``{name: float}``).

        The no-argument attribute form of :meth:`final_named` -- ``sol.final``."""
        return self.final_named()

    def units_named(self, species: str) -> str:
        """Return the declared units of a species (for axis/column labels).

        Convenience for plotting and tabulating results without re-deriving
        units by string-matching species names. Equivalent to
        ``self.model.units_of(species)``.
        """
        return self.model.units_of(species)

    @property
    def time_unit(self) -> str | None:
        """The time unit of ``self.t`` (``"s"``, ``"d"``, ... or ``None``).

        Defaults to the model's native unit (:attr:`CompiledModel.time_unit`),
        surfaced here so a plot or table can label the time axis unambiguously.
        When the solve was called with an explicit ``time_unit=`` the result times
        are reported in *that* unit, and this returns it."""
        override = getattr(self, "_requested_time_unit", None)
        return override if override is not None else self.model.time_unit

    def _table_index(self) -> tuple[str, jnp.ndarray]:
        """Return ``(name, array)`` for the dataframe index. Time by default;
        space-indexed solutions (PFR) override this."""
        return "t", self.t

    def _independent_axis_label(self) -> str:
        """Label for the plot's independent axis. Time (with the model's
        time unit) by default; a space-indexed PFR overrides this."""
        unit = self.time_unit
        return f"time [{unit}]" if unit else "time"


class PlottableSolutionMixin:
    """Mixin: matplotlib rendering of a solution's species trajectories.

    Composed onto a solution dataclass **alongside** :class:`_HasNamedSpecies`,
    whose ``C_named`` / ``units_named`` accessors and ``_table_index`` /
    ``_independent_axis_label`` it reads. Kept separate from the pure accessor so
    a change to the plotting API does not touch the numeric ``C_named`` every
    reactor depends on, and so the optional matplotlib dependency stays isolated.
    """

    def plot(self, species=None, *, ax=None, **kwargs):
        """Plot one or more species against the independent axis (time, or axial
        position for a PFR).

        A thin wrapper over matplotlib so "plot SNH over time" needs no manual
        ``C_named`` / unit / axis-label boilerplate. The x-axis is labelled with
        the model's time unit (or position for a PFR), and a single-species
        plot labels the y-axis with that species' units.

        Parameters
        ----------
        species : str or iterable of str, optional
            Species to plot. A single name plots one line; an iterable plots
            several with a legend; ``None`` (default) plots every species.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If ``None``, a new figure/axes is created.
        **kwargs
            Forwarded to ``ax.plot`` (e.g. ``lw``, ``ls``, ``color``).

        Returns
        -------
        matplotlib.axes.Axes
            The axes drawn on (for further customisation / saving).

        Raises
        ------
        ImportError
            If matplotlib is not installed (optional dependency; install with
            ``pip install aquakin[plot]``).
        KeyError
            If a species name is unknown (with a "did you mean?" hint).
        """
        plt = require_matplotlib()
        names = (
            [species]
            if isinstance(species, str)
            else list(self.model.species)
            if species is None
            else list(species)
        )
        if ax is None:
            _, ax = plt.subplots()
        _, x = self._table_index()
        x = np.asarray(x)
        for sp in names:
            ax.plot(x, np.asarray(self.C_named(sp)), label=sp, **kwargs)
        ax.set_xlabel(self._independent_axis_label())
        if len(names) == 1:
            ax.set_ylabel(f"{names[0]} [{self.units_named(names[0])}]")
        else:
            ax.set_ylabel("concentration")
            ax.legend()
        return ax


class ExportableSolutionMixin:
    """Mixin: pandas / CSV export of a solution's species trajectories.

    Composed onto a solution dataclass **alongside** :class:`_HasNamedSpecies`,
    whose ``C`` / ``model`` / ``_table_index`` it reads. Kept separate from the
    pure accessor so a change to the CSV schema or the optional pandas dependency
    does not touch the core ``C_named`` accessor.
    """

    def to_dataframe(self, *, units_in_columns: bool = False):
        """Return the solution as a pandas ``DataFrame``.

        One row per recorded point, one column per species (in model
        ordering), indexed by the independent axis (time ``t`` for batch /
        track / biofilm solutions, axial position ``x`` for a PFR).

        Parameters
        ----------
        units_in_columns : bool, optional
            If ``True``, append ``" [unit]"`` to each species column label
            (e.g. ``"SNH [g_N/m³]"``). If ``False`` (default), columns are bare
            species names and the per-species units are stored in
            ``df.attrs["units"]`` instead, which keeps columns selectable by
            species name.

        Returns
        -------
        pandas.DataFrame

        Raises
        ------
        ImportError
            If pandas is not installed (it is an optional dependency; install
            with ``pip install aquakin[dataframe]``).
        """
        model = self.model
        columns = [(sp, self.C[:, j]) for j, sp in enumerate(model.species)]
        units = {sp: model.units_of(sp) for sp in model.species}
        name, index = self._table_index()
        return build_dataframe(
            index,
            columns,
            index_name=name,
            units=units,
            units_in_columns=units_in_columns,
        )

    def to_csv(self, path_or_buf=None, *, units_in_columns: bool = True, **kwargs):
        """Write the solution to CSV (delegates to :meth:`to_dataframe`).

        Parameters
        ----------
        path_or_buf : str or path or file-like, optional
            Destination passed to ``DataFrame.to_csv``. If ``None``, the CSV is
            returned as a string.
        units_in_columns : bool, optional
            Defaults to ``True`` here (unlike :meth:`to_dataframe`): a CSV
            cannot carry ``df.attrs``, so the units are embedded in the column
            headers by default so the written file is self-describing.
        **kwargs
            Forwarded to ``pandas.DataFrame.to_csv``.
        """
        return self.to_dataframe(units_in_columns=units_in_columns).to_csv(path_or_buf, **kwargs)
