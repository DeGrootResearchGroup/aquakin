"""Scenario comparison and standardized KPI tabulation.

Two closely related tabulation workflows:

* :func:`compare_scenarios` -- run a model under several named input sets (design
  options, operating points) and tabulate the resulting KPIs side by side. Like
  :func:`aquakin.monte_carlo`, it takes an ``fn(x) -> output`` callback that runs
  the solve itself.
* :func:`kpi_comparison` -- the standardized-report companion: it does **no
  solve**, only assembles a side-by-side table from heterogeneous, already-computed
  report objects (a ``BSM2Evaluation``, ``CarbonFootprint``, ``OperatingCost``, or
  any object exposing a ``kpis()`` mapping, or a plain ``{name: value}`` dict).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from aquakin.integrate._qmc import _eval_fn_over, _resolve_output_names

# --- Scenario comparison -----------------------------------------------------


@dataclass
class ScenarioComparison:
    """Result of :func:`compare_scenarios`: KPIs per named scenario.

    Attributes
    ----------
    scenario_names : list[str]
        The scenarios, row order of ``outputs`` / ``inputs``.
    input_names, output_names : list[str]
        Names of the input and output columns.
    inputs : np.ndarray
        ``(n_scenarios, d)`` input vector used for each scenario.
    outputs : np.ndarray
        ``(n_scenarios, m)`` outputs.
    """

    scenario_names: list[str]
    input_names: list[str]
    output_names: list[str]
    inputs: np.ndarray
    outputs: np.ndarray

    def _col(self, name: str) -> np.ndarray:
        if name not in self.output_names:
            raise KeyError(f"unknown output '{name}'; have {self.output_names}.")
        return self.outputs[:, self.output_names.index(name)]

    def output_named(self, name: str) -> np.ndarray:
        """The output ``name`` across scenarios, shape ``(n_scenarios,)``."""
        return self._col(name)

    def best(self, output: str, *, minimize: bool = True) -> str:
        """The scenario name with the lowest (or highest) value of ``output``."""
        col = self._col(output)
        idx = int(np.argmin(col) if minimize else np.argmax(col))
        return self.scenario_names[idx]

    def table(self) -> str:
        """A human-readable KPI table, one row per scenario."""
        cols = ["scenario"] + list(self.output_names)
        rows = [cols]
        for i, name in enumerate(self.scenario_names):
            rows.append(
                [name] + [f"{self.outputs[i, k]:.4g}" for k in range(len(self.output_names))]
            )
        w = [max(len(r[c]) for r in rows) for c in range(len(cols))]
        return "\n".join("  ".join(r[c].ljust(w[c]) for c in range(len(cols))) for r in rows)


def compare_scenarios(
    fn: Callable,
    scenarios: dict,
    *,
    input_names: Sequence[str],
    baseline: Optional[Sequence[float]] = None,
    output_names: Optional[Sequence[str]] = None,
    batched: bool = True,
) -> ScenarioComparison:
    """Run ``fn`` under several named scenarios and tabulate the outputs.

    Parameters
    ----------
    fn : callable
        ``fn(x) -> output`` as in :func:`monte_carlo` / :func:`aquakin.dgsm`.
    scenarios : dict
        ``name -> overrides`` where ``overrides`` is either a full input vector
        (length ``d``) or a mapping ``{input_name: value}`` applied on top of
        ``baseline`` (so a scenario only states what it changes). An empty
        mapping ``{}`` is the baseline itself.
    input_names : sequence of str
        Names of the ``d`` inputs (defines the vector order and the override
        keys).
    baseline : sequence of float, optional
        The nominal input vector that mapping-style overrides modify. Required if
        any scenario uses ``{input_name: value}`` overrides; defaults to zeros.
    output_names : sequence of str, optional
        Output column names.
    batched : bool
        vmap the scenarios (default) or evaluate one at a time.

    Returns
    -------
    ScenarioComparison
    """
    input_names = list(input_names)
    d = len(input_names)
    base = np.zeros(d) if baseline is None else np.asarray(baseline, dtype=float)
    if base.shape != (d,):
        raise ValueError(f"baseline must have shape ({d},); got {base.shape}.")
    idx = {n: i for i, n in enumerate(input_names)}

    names = list(scenarios.keys())
    X = np.empty((len(names), d))
    for r, name in enumerate(names):
        ov = scenarios[name]
        if isinstance(ov, dict):
            x = base.copy()
            for k, v in ov.items():
                if k not in idx:
                    raise KeyError(
                        f"scenario '{name}' overrides unknown input '{k}'; "
                        f"inputs are {input_names}."
                    )
                x[idx[k]] = float(v)
        else:
            x = np.asarray(ov, dtype=float)
            if x.shape != (d,):
                raise ValueError(f"scenario '{name}' vector must have shape ({d},); got {x.shape}.")
        X[r] = x

    Y, finite = _eval_fn_over(fn, X, batched)
    if not finite.all():
        bad = [names[i] for i in range(len(names)) if not finite[i]]
        raise ValueError(f"scenario(s) gave a non-finite output: {bad}.")
    return ScenarioComparison(
        scenario_names=names,
        input_names=input_names,
        output_names=_resolve_output_names(output_names, Y.shape[1]),
        inputs=X,
        outputs=Y,
    )


# --- Standardized KPI comparison ---------------------------------------------


@dataclass
class KPIComparison:
    """A side-by-side KPI table over several named results.

    The standardized-report companion to :func:`compare_scenarios`: where that
    runs a model and tabulates a fixed output *vector*, this assembles a table
    from heterogeneous **report objects** (a :class:`BSM2Evaluation`, a
    :class:`CarbonFootprint`, an :class:`OperatingCost`, or any object exposing a
    ``kpis()`` mapping -- or a plain ``{name: value}`` dict) already computed per
    scenario. The KPI columns are the union of every report's keys, in
    first-seen order; a KPI a given report does not provide is left blank.

    Attributes
    ----------
    names : list[str]
        The result names (table rows).
    kpi_names : list[str]
        The KPI labels (table columns), union over all results.
    values : dict
        ``name -> {kpi: value}`` for every result.
    """

    names: list[str]
    kpi_names: list[str]
    values: dict

    def column(self, kpi: str) -> dict:
        """The ``{name: value}`` map for one KPI across results."""
        if kpi not in self.kpi_names:
            raise KeyError(f"unknown KPI '{kpi}'; have {self.kpi_names}.")
        return {n: self.values[n].get(kpi, float("nan")) for n in self.names}

    def best(self, kpi: str, *, minimize: bool = True) -> str:
        """The result name with the lowest (or highest) value of ``kpi``."""
        col = self.column(kpi)
        finite = {n: v for n, v in col.items() if v == v}  # drop NaNs
        if not finite:
            raise ValueError(f"KPI '{kpi}' has no finite value across results.")
        return (min if minimize else max)(finite, key=finite.get)

    def table(self) -> str:
        """A human-readable KPI table, one column per result."""
        rows = [["KPI", *self.names]]
        for kpi in self.kpi_names:
            row = [kpi]
            for n in self.names:
                v = self.values[n].get(kpi)
                row.append("" if v is None else f"{v:.4g}")
            rows.append(row)
        w = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
        return "\n".join("  ".join(r[c].ljust(w[c]) for c in range(len(r))) for r in rows)

    def __str__(self) -> str:
        return self.table()


def _kpis_of(report) -> dict:
    """Extract a ``{kpi: value}`` mapping from a report object or plain dict."""
    if isinstance(report, dict):
        return dict(report)
    kpis = getattr(report, "kpis", None)
    if callable(kpis):
        return dict(kpis())
    raise TypeError(
        f"a KPI report must be a dict or expose a kpis() method; got {type(report).__name__}."
    )


def kpi_comparison(reports: dict) -> KPIComparison:
    """Tabulate KPIs from several named report objects side by side.

    Parameters
    ----------
    reports : dict
        ``name -> report``, where each report is a result object exposing a
        ``kpis()`` method (:class:`BSM2Evaluation`, :class:`CarbonFootprint`,
        :class:`OperatingCost`, ...) or a plain ``{kpi: value}`` mapping. The KPI
        columns are the union of every report's keys, in first-seen order.

    Returns
    -------
    KPIComparison

    Examples
    --------
    >>> kpi_comparison({
    ...     "baseline": evaluation_a,
    ...     "low-DO":   evaluation_b,
    ... }).table()  # doctest: +SKIP
    """
    names = list(reports.keys())
    per_name = {n: _kpis_of(reports[n]) for n in names}
    kpi_names: list = []
    for n in names:
        for k in per_name[n]:
            if k not in kpi_names:
                kpi_names.append(k)
    return KPIComparison(names=names, kpi_names=kpi_names, values=per_name)
