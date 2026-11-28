"""Sampling the posterior of a plant calibration: :meth:`Plant.sample_posterior`.

The plant end of :func:`aquakin.sample_posterior`. The sampler, its target
density and its diagnostics are the reactor's (``integrate/calibration/
sampling.py``, reached through ``integrate/seam.py``); only the forward model
differs, and it is the one :meth:`Plant.calibrate` fitted through.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aquakin.integrate.calibration.config import SamplerConfig
from aquakin.integrate.seam import _check_sampling_result, _run_sampler
from aquakin.plant.calibrate import _PlantForwardModel

if TYPE_CHECKING:
    from aquakin.integrate.calibration.result import CalibrationResult, PosteriorSamples
    from aquakin.plant.plant import Plant


def sample_posterior_plant(
    plant: Plant,
    result: CalibrationResult,
    *,
    config: SamplerConfig = SamplerConfig(),
    prior_space: str = "transformed",
    progress=None,
) -> PosteriorSamples:
    """Draw from the posterior of a plant calibration by Markov chain Monte Carlo.

    The plant counterpart of :func:`aquakin.sample_posterior`, with the same
    sampler, density and diagnostics: a random-walk Metropolis over the
    coordinates the fit ran in, started near the MAP with the Laplace
    covariance as its proposal, on the posterior the fit maximized. As there,
    ``prior_space`` sets whether the prior of a transformed parameter is flat
    in its transformed coordinate (the default, the density the fit and its
    Laplace approximation use) or in its physical value. Fitted initial pools
    are sampled jointly with the parameters. Each step costs one plant solve
    per chain.

    Parameters
    ----------
    plant : Plant
        The plant the result was calibrated on.
    result : CalibrationResult
        A fit from :meth:`Plant.calibrate` on this plant with ``loss="nll"``
        and a Laplace posterior (``laplace=True``, or a
        :class:`~aquakin.LaplaceConfig`; with ``include_ic=True`` when initial
        pools were fitted). The solves use the fit's own settings (its
        ``y0``, observed streams, integrator and time unit).
    config : SamplerConfig, optional
        Chains, draws, warm-up and seed. Default ``SamplerConfig()``.
    prior_space : {"transformed", "physical"}, optional
        The space in which the posterior is a density, as for
        :func:`aquakin.sample_posterior`. Default ``"transformed"``.
    progress : callable, optional
        Called after each step of the chains, as for
        :func:`aquakin.sample_posterior`.

    Returns
    -------
    PosteriorSamples
        The draws, laid out as :meth:`CalibrationResult.posterior_draws` (full
        plant parameter vectors), with the R-hat, effective sample size and
        acceptance rate.

    Raises
    ------
    ValueError
        If ``result`` did not come from :meth:`Plant.calibrate` on this plant
        with ``loss="nll"`` and a Laplace posterior, or ``prior_space`` is not
        ``"transformed"`` or ``"physical"``.
    """
    _check_sampling_result(result)
    fm = result._fit_record.forward_model
    if not isinstance(fm, _PlantForwardModel):
        raise ValueError(
            "This result comes from aquakin.calibrate; sample it with "
            "aquakin.sample_posterior(result, reactor)."
        )
    if fm.plant is not plant:
        raise ValueError("This result was calibrated on a different plant.")
    return _run_sampler(result, fm, config, prior_space, progress)
