"""Settings of the lean loop that the plant-sensitivity scripts compare.

Shared by ``profile_plant_sensitivity.py --variants`` and
``probe_step_rejections.py --settings``. A spec is a comma-separated list of:

- ``default`` -- nothing (the shipped settings);
- ``plain`` / ``anderson:DEPTH`` -- the Anderson depth of the stage iteration
  (``_ANDERSON_DEPTH``; ``plain`` is depth 0);
- ``kappa:VALUE`` -- the stage convergence tolerance (``_KAPPA``);
- ``noerrcon`` -- the sensitivity left out of the error test
  (``solve_sensitivity(sens_error_control=False)``, CVODES ``errconS = false``);
- ``FIELD:VALUE`` -- a numeric ``StepControl`` field (``safety``, ``pi_beta``,
  ``maxfac``, ``minfac``, ``theta_target``).

Unset settings keep their shipped values. All but ``noerrcon`` are module
settings of ``aquakin.integrate.forward_solve`` (and the ``StepControl`` the
plant builds), read when a solve is compiled: apply a spec before a plant's
first solve. ``noerrcon`` is a ``solve_sensitivity`` argument: pass
``solve_kwargs`` to every call.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import aquakin.integrate.forward_solve as forward_solve

_STEP_CONTROL = forward_solve.StepControl
_SHIPPED = (forward_solve._ANDERSON_DEPTH, forward_solve._KAPPA)
_NUMERIC_FIELDS = ("safety", "pi_beta", "maxfac", "minfac", "theta_target")


@dataclass(frozen=True)
class LeanSettings:
    """One parsed spec (see the module docstring)."""

    spec: str
    anderson_depth: int
    kappa: float
    sens_error_control: bool
    control: tuple[tuple[str, object], ...]

    @classmethod
    def parse(cls, spec: str) -> LeanSettings:
        depth, kappa = _SHIPPED
        sens_error = True
        control = {}
        for item in spec.split(","):
            kind, _, value = item.partition(":")
            if kind == "default":
                pass
            elif kind == "plain":
                depth = 0
            elif kind == "anderson":
                depth = int(value)
            elif kind == "kappa":
                kappa = float(value)
            elif kind == "noerrcon":
                sens_error = False
            elif kind in _NUMERIC_FIELDS:
                control[kind] = float(value)
            else:
                raise ValueError(f"unknown lean-loop setting {item!r}")
        return cls(spec, depth, kappa, sens_error, tuple(control.items()))

    @property
    def solve_kwargs(self) -> dict:
        """The ``solve_sensitivity`` keyword arguments of this spec."""
        return {} if self.sens_error_control else {"sens_error_control": False}

    @property
    def tag(self) -> str:
        """The spec as a file-name-safe tag."""
        return self.spec.replace(":", "").replace(",", "+")

    def apply(self) -> None:
        """Set the module settings (and the plant's ``StepControl``) to this spec."""
        forward_solve._ANDERSON_DEPTH = self.anderson_depth
        forward_solve._KAPPA = self.kappa
        forward_solve.StepControl = functools.partial(_STEP_CONTROL, **dict(self.control))
