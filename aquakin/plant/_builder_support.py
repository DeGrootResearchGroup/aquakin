"""Shared construction helpers for the plant builders (BSM1 / BSM2 / A2O).

The three flowsheet builders (:func:`~aquakin.plant.bsm.build_bsm1`,
:func:`~aquakin.plant.bsm.build_bsm2`, :func:`~aquakin.plant.a2o.build_a2o`) share
several boilerplate steps: seeding the activated-sludge reactor conditions from
the model's declared defaults, computing the constant recycle-pump flows,
selecting the secondary clarifier, and registering the canonical semantic
streams. They live here once, with the shared rationale carried in one place, so
each builder reads as flowsheet topology rather than copy-paste.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aquakin.plant.clarifier import IdealClarifier
from aquakin.plant.takacs import TakacsClarifier

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.model import CompiledModel
    from aquakin.plant.plant import Plant


def reactor_conditions(model: CompiledModel) -> dict[str, float]:
    """The scalar ``conditions=`` dict for the activated-sludge reactors.

    Each condition the model requires, set to the model's declared default via the
    public :meth:`~aquakin.core.model.CompiledModel.condition_defaults` accessor
    (so the builders do not reach into the private ``_condition_defaults``).
    """
    defaults = model.condition_defaults()
    return {name: defaults[name] for name in model.conditions_required}


def recycle_pump_flows(*, internal_ratio, ras_ratio, Q_design, wastage):
    """The constant volumetric recycle-pump flows shared by the BSM / A2O layouts.

    Returns ``(Qa, Qr, Qw, Q_underflow)`` -- internal (nitrate) recycle, RAS,
    wastage, and the secondary-clarifier underflow ``Qr + Qw``. These are **fixed
    volumetric setpoints** off the design flow ``Q_design``, not fractions of
    throughput: a fixed fraction makes the recycle-flow loop gain near-singular off
    the design influent, so the plant blows up under dynamic flow (see
    :class:`~aquakin.plant.mixer.SetpointSplitter`). The clarifier effluent is the
    free remainder ``Q_e = Q_f - Q_underflow``.
    """
    Qa = internal_ratio * Q_design
    Qr = ras_ratio * Q_design
    Qw = wastage
    return Qa, Qr, Qw, Qr + Qw


def add_secondary_clarifier(
    plant: Plant,
    *,
    model: CompiledModel,
    underflow_Q,
    use_takacs: bool,
    takacs_kwargs: dict,
    ideal_kwargs: dict | None = None,
) -> None:
    """Add the ``"clarifier"`` unit, selecting the settler model.

    ``use_takacs`` picks the full Takacs 1-D layered secondary clarifier (the BSM
    reference); otherwise the fast, stateless ~99.8%-capture
    :class:`~aquakin.plant.clarifier.IdealClarifier` (its ``capture_efficiency``
    default). Both expose the same ``overflow`` / ``underflow`` ports, so the rest
    of the flowsheet graph is identical either way. ``takacs_kwargs`` /
    ``ideal_kwargs`` carry the per-plant settler geometry / options; ``name``,
    ``model`` and ``underflow_Q`` are shared.
    """
    if use_takacs:
        plant.add_unit(
            TakacsClarifier(name="clarifier", model=model, underflow_Q=underflow_Q, **takacs_kwargs)
        )
    else:
        plant.add_unit(
            IdealClarifier(
                name="clarifier", model=model, underflow_Q=underflow_Q, **(ideal_kwargs or {})
            )
        )


def register_recycle_streams(
    plant: Plant, *, internal_recycle: str, ras: str, wastage: str, effluent: str | None = None
) -> None:
    """Register the canonical semantic stream names shared by the BSM / A2O builders.

    ``effluent`` (defaults to :attr:`Plant.effluent_endpoint`), ``internal_recycle``,
    ``ras`` and ``wastage`` -- so ``plant.stream(sol, "ras")`` /
    ``plant.list_streams()`` read by role rather than by ``"unit.port"``. Each value
    is the source endpoint string.
    """
    plant.register_stream("effluent", effluent or plant.effluent_endpoint)
    plant.register_stream("internal_recycle", internal_recycle)
    plant.register_stream("ras", ras)
    plant.register_stream("wastage", wastage)
