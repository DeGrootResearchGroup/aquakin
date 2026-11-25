"""The solution-object mixins (species by name, plotting, tabular export) and
the optional pandas / matplotlib imports they need.

Not part of the public API. ``aquakin.integrate._common`` re-exports the names
defined before the module split; import later additions (``name_list``) from
here.
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


def name_list(names) -> list[str]:
    """Normalize a by-name selection to a list of names.

    A bare ``str`` is one name, not an iterable of one-character names, so
    ``name_list("SNH")`` is ``["SNH"]``; any other iterable of names is listed
    as given. Every accessor that takes "a name or several" reads its argument
    through this, so they all treat a single string the same way.

    Parameters
    ----------
    names : str or iterable of str
        One name, or several.

    Returns
    -------
    list of str
    """
    return [names] if isinstance(names, str) else list(names)


def requested_units(names, units) -> dict[str, str]:
    """``{species: unit}`` for the species a reader should convert.

    Parameters
    ----------
    names : list of str
        The species being read.
    units : str, dict[str, str] or None
        One unit for every species read, a ``{species: unit}`` dict naming
        some of them (the rest stay in the model's units), or ``None``.

    Raises
    ------
    ValueError
        If the dict names a species that is not being read.
    """
    if units is None:
        return {}
    if isinstance(units, str):
        return dict.fromkeys(names, units)
    extra = sorted(set(units) - set(names))
    if extra:
        raise ValueError(
            f"units names {extra}, which are not among the species read ({list(names)})."
        )
    return dict(units)


def species_in_unit(model: CompiledModel, species: str, values, unit: str | None):
    """``values`` of ``species`` (in ``model``'s declared unit) expressed in ``unit``.

    Only the scale changes: ``g/m3`` to ``mg/L`` is 1, ``g_COD/m3`` to
    ``kgCOD/m3`` is 0.001. ``values`` is returned untouched when ``unit`` is
    ``None`` or the factor is exactly 1, and multiplied otherwise, so the
    result stays differentiable.

    Raises
    ------
    ValueError
        If ``unit`` measures another quantity than the model's unit (moles
        against grams, g N against g COD), or either cannot be read.
    """
    from aquakin.utils.units import conversion_factor

    if unit is None:
        return values
    native = model.units_of(species)
    try:
        factor = conversion_factor(native, unit)
    except ValueError as exc:
        raise ValueError(f"Cannot give '{species}' in '{unit}': {exc}") from None
    if factor is None:
        raise ValueError(
            f"Cannot give '{species}' in '{unit}': model '{model.name}' declares it in "
            f"'{native}', and one of the two cannot be read."
        )
    return values if factor == 1.0 else values * factor


def species_columns(model: CompiledModel, C, units=None):
    """The ``(name, values)`` columns and ``{name: unit}`` labels of a table of
    every species of ``model``, with ``C`` shape ``(n, n_species)``, in the
    requested ``units`` (see :func:`requested_units`)."""
    want = requested_units(list(model.species), units)
    columns = [
        (sp, species_in_unit(model, sp, C[:, j], want.get(sp)))
        for j, sp in enumerate(model.species)
    ]
    labels = {sp: want.get(sp, model.units_of(sp)) for sp in model.species}
    return columns, labels


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

    def C_named(self, species: str, *, units: str | None = None) -> jnp.ndarray:
        """Return the trajectory of a single species by name.

        Parameters
        ----------
        species : str
            The species name.
        units : str, optional
            Unit to return it in, e.g. ``"mg/L"`` or ``"kgCOD/m3"``. Converted
            from the unit the model declares for the species (see
            :meth:`units_named`); only the scale changes, so a unit of another
            quantity raises ``ValueError``. ``None`` (default) returns the
            model's unit.

        Returns
        -------
        jnp.ndarray
            The trajectory, shape ``(n_points,)``.
        """
        if species not in self.model.species_index:
            suffix = did_you_mean(species, self.model.species)
            raise KeyError(f"Unknown species '{species}'. Available: {self.model.species}.{suffix}")
        values = self.C[:, self.model.species_index[species]]
        return species_in_unit(self.model, species, values, units)

    def C_named_many(self, species, *, units=None) -> dict[str, jnp.ndarray]:
        """Trajectories of several species by name, as ``{name: array}``.

        The multi-species companion to :meth:`C_named` -- read a handful of
        species in one call (``sol.C_named_many(["SNH", "SNO", "SO"])``) instead
        of a separate slice each. Each value has the same shape ``C_named``
        returns. An unknown name raises the same hinted ``KeyError``.

        Parameters
        ----------
        species : str or iterable of str
            The species names. A single string is one name, giving the same
            one-entry dict as a one-element list.
        units : str or dict[str, str], optional
            One unit for every species read, or ``{species: unit}`` for some of
            them (the rest stay in the model's units); see :meth:`C_named`.

        Returns
        -------
        dict[str, jnp.ndarray]
            ``{name: trajectory}`` in the order given.
        """
        names = name_list(species)
        want = requested_units(names, units)
        return {sp: self.C_named(sp, units=want.get(sp)) for sp in names}

    def final_named(self, species=None, *, units=None) -> dict[str, float]:
        """Values at the **last** recorded point, as ``{name: float}``.

        The reporting shortcut for a steady-state / end-of-run value: instead of
        ``float(sol.C_named("SNH")[-1])`` per species, ``sol.final_named(["SNH",
        "SNO"])`` returns them in one dict. Values are plain Python floats (this
        is a post-processing read on an already-solved solution -- use
        ``C_named(sp)[-1]`` if you need a differentiable last value).

        Parameters
        ----------
        species : str or iterable of str, optional
            The species names. A single string is one name
            (``final_named("SNH")`` equals ``final_named(["SNH"])``). ``None``
            (default) returns every model species.
        units : str or dict[str, str], optional
            One unit for every species read, or ``{species: unit}`` for some of
            them; see :meth:`C_named`.

        Returns
        -------
        dict[str, float]
            ``{name: last value}`` in the order given.
        """
        names = list(self.model.species) if species is None else name_list(species)
        want = requested_units(names, units)
        return {sp: float(self.C_named(sp, units=want.get(sp))[-1]) for sp in names}

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
        names = list(self.model.species) if species is None else name_list(species)
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

    def to_dataframe(self, *, units_in_columns: bool = False, units=None):
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
        units : str or dict[str, str], optional
            Units to tabulate in: one for every species, or ``{species: unit}``
            for some of them (the rest stay in the model's units). The values
            are converted as by :meth:`C_named`, and the column labels and
            ``df.attrs["units"]`` give the units used.

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
        columns, units = species_columns(model, self.C, units)
        name, index = self._table_index()
        return build_dataframe(
            index,
            columns,
            index_name=name,
            units=units,
            units_in_columns=units_in_columns,
        )

    def to_csv(self, path_or_buf=None, *, units_in_columns: bool = True, units=None, **kwargs):
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
        units : str or dict[str, str], optional
            Units to write the species in (see :meth:`to_dataframe`).
        **kwargs
            Forwarded to ``pandas.DataFrame.to_csv``.
        """
        df = self.to_dataframe(units_in_columns=units_in_columns, units=units)
        return df.to_csv(path_or_buf, **kwargs)
