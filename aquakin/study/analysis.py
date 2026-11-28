"""What a study's analyses share, whatever they study.

A plant study and a reactor study call different analyses (``calibrate_plant``
or ``calibrate``, ...), but read their observations, set up their fit, and write
their Fisher information and profile with the functions here, so the two
targets give the same files.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from aquakin.study.common import (
    DRAWS_FILE,
    FISHER_FILE,
    FISHER_SUMMARY_FILE,
    PARAMETERS_FILE,
    POSTERIOR_FILE,
    PROFILE_FILE,
    PROFILE_SUMMARY_FILE,
    StudyError,
)
from aquakin.utils.result_files import write_matrix, write_named_values, write_yaml

if TYPE_CHECKING:
    from aquakin.study.study import _Study


def _fit_sigma(study: _Study):
    """The fit's ``sigma``: ``"estimate"``, one error for every observation, or an
    array of one per channel (broadcast over the times; a list would mean one
    per dataset)."""
    import numpy as np

    if study.estimate_sigma:
        return "estimate"
    if isinstance(study.sigma, list):
        return np.asarray(study.sigma, dtype=float)
    return study.sigma


def _flattened(ref, name: str, base_dir: Path) -> str:
    """A model's YAML with its whole ``extends`` chain merged in."""
    from importlib.resources import files

    from aquakin.schema.loader import _resolve_inheritance, parse_model_yaml

    if ref["source"] == "shipped":
        text = (files("aquakin.models") / f"{name}.yaml").read_text()
        base = None
    else:
        path = Path(ref["source"])
        path = path if path.is_absolute() else base_dir / path
        text, base = path.read_text(), path.parent
    data = _resolve_inheritance(parse_model_yaml(text), name, base, seen=())
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def _fit_options(study: _Study, start=None) -> dict:
    """The settings of a study's fit that do not depend on what it fits: the
    transforms, the likelihood with its measurement error, the priors and the
    optimizer. With ``start`` (a full parameter vector), the optimizer starts
    there alone, in place of the study's starts: how a posterior study re-enters
    a saved fit at its optimum without changing what is fitted."""
    from aquakin import OptimizerConfig

    if start is None:
        optimizer = OptimizerConfig(n_starts=study.n_starts, max_iter=study.max_iter)
    else:
        optimizer = OptimizerConfig(starts=[start], max_iter=study.max_iter)
    return dict(
        transforms=dict(study.transforms) or None,
        loss="nll",
        sigma=_fit_sigma(study),
        use_priors=study.use_priors,
        optimizer=optimizer,
    )


def _read_observations(path: Path, names: list[str], where: str):
    """The named columns of a data table.

    Parameters
    ----------
    path : pathlib.Path
        A table written as :func:`~aquakin.utils.result_files.write_trajectory`
        writes one: a ``t`` column, then one column per name.
    names : list of str
        The columns to read, in order.
    where : str
        The setting that names the file, for messages.

    Returns
    -------
    t : numpy.ndarray
        ``(n_t,)``.
    observations : numpy.ndarray
        ``(n_t, len(names))``.

    Raises
    ------
    StudyError
        If the file is missing, is not such a table, or lacks a column.
    """
    import numpy as np

    from aquakin.utils.result_files import read_trajectory

    if not path.is_file():
        raise StudyError(f"{where}: {path} does not exist")
    try:
        t, columns = read_trajectory(path)
    except ValueError as exc:
        raise StudyError(f"{where}: {exc}") from None
    missing = [n for n in names if n not in columns]
    if missing:
        raise StudyError(
            f"{where}: {path.name} has no column {', '.join(missing)} (its columns: "
            f"{', '.join(columns) or 'none'})"
        )
    return t, np.stack([columns[n] for n in names], axis=1)


def _std(fim) -> dict[str, float] | None:
    """A Fisher information's standard deviations by name, ``None`` when it is
    singular."""
    import numpy as np

    try:
        return fim.std()
    except np.linalg.LinAlgError:
        return None


def _write_fisher_files(out: Path, fim, parameters: dict) -> list[Path]:
    """Write a Fisher information's ``fisher.csv`` and ``summary.yaml`` (values,
    transforms, standard deviations or ``None`` when it is singular, the four
    design criteria), and the parameters it was evaluated at."""
    import numpy as np

    from aquakin import design_criterion

    criteria = {}
    for c in ("D", "A", "E", "modified_E"):
        try:
            criteria[c] = float(design_criterion(fim, c))
        except (np.linalg.LinAlgError, ValueError):
            criteria[c] = None
    summary = {
        "values": dict(fim.values),
        "transforms": dict(zip(fim.names, fim.transforms)),
        "std": _std(fim),
        "criteria": criteria,
    }
    return [
        write_matrix(out / FISHER_FILE, fim.matrix, fim.names),
        write_yaml(out / FISHER_SUMMARY_FILE, summary),
        write_named_values(out / PARAMETERS_FILE, parameters),
    ]


def _write_profile_files(out: Path, study: _Study, prof) -> list[Path]:
    """Write a profile likelihood's ``profile.csv`` (the grid, the loss and its
    rise from the best) and ``profile.yaml`` (the best value and the interval)."""
    import numpy as np

    from aquakin.utils.result_files import write_table

    written = [
        write_table(
            out / PROFILE_FILE,
            {
                study.profile: list(prof.grid),
                "loss": [None if not np.isfinite(v) else v for v in prof.loss],
                "delta_loss": [None if not np.isfinite(v) else v for v in prof.delta_loss],
            },
        )
    ]
    lo, hi = prof.ci
    summary = {
        "parameter": study.profile,
        "mle": None if not np.isfinite(prof.mle) else float(prof.mle),
        "ci": [None if lo is None else float(lo), None if hi is None else float(hi)],
        "delta": study.delta,
    }
    written.append(write_yaml(out / PROFILE_SUMMARY_FILE, summary))
    return written


def _write_posterior_files(out: Path, study: _Study, samples, fit) -> list[Path]:
    """Write a posterior study's ``draws.csv`` (every draw of every chain, by
    parameter name) and ``posterior.yaml`` (the sampler's settings, each
    parameter's optimum, mean, standard deviation, 5 / 50 / 95 % quantiles,
    R-hat and effective sample size, and each chain's acceptance rate)."""
    import numpy as np

    from aquakin.utils.result_files import write_table

    n_chains, n_draws = samples.chains.shape[:2]
    names = list(samples.parameter_names)
    values = {n: np.asarray(samples.params_named[n], dtype=float) for n in names}
    draws = {
        "chain": [c for c in range(n_chains) for _ in range(n_draws)],
        "draw": [d for _ in range(n_chains) for d in range(n_draws)],
        **{n: [float(x) for x in values[n]] for n in names},
    }
    parameters = {}
    for n in names:
        v = values[n]
        q05, q50, q95 = (float(x) for x in np.quantile(v, [0.05, 0.5, 0.95]))
        parameters[n] = {
            "optimum": float(fit.params_named[n]),
            "mean": float(v.mean()),
            "std": float(v.std(ddof=1)),
            "q05": q05,
            "q50": q50,
            "q95": q95,
            "r_hat": float(samples.r_hat[n]),
            "ess_bulk": float(samples.ess_bulk[n]),
        }
    summary = {
        "fit": study.fit,
        "n_chains": n_chains,
        "n_draws": n_draws,
        "n_warmup": study.n_warmup,
        "seed": study.seed,
        "prior_space": study.prior_space,
        "parameters": parameters,
        "acceptance_rate": [float(a) for a in samples.acceptance_rate],
    }
    return [write_table(out / DRAWS_FILE, draws), write_yaml(out / POSTERIOR_FILE, summary)]


def _sampler(study: _Study):
    """The sampler settings of a posterior study."""
    from aquakin import SamplerConfig

    return SamplerConfig(
        n_chains=study.n_chains, n_draws=study.n_draws, n_warmup=study.n_warmup, seed=study.seed
    )
