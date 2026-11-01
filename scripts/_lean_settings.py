"""Settings of the lean loop that the plant-sensitivity scripts compare.

Shared by ``profile_plant_sensitivity.py --variants`` and
``probe_step_rejections.py --settings``. A spec is a comma-separated list of:

- ``default`` -- nothing (the shipped settings);
- ``plain`` / ``anderson:DEPTH`` -- the Anderson depth of the stage iteration
  (``_ANDERSON_DEPTH``; ``plain`` is depth 0);
- ``value`` -- the stage convergence test on the stage-value increment
  ``h*G*dk`` (``_STAGE_TEST_IN_VALUE_UNITS``);
- ``noerrcon`` -- the sensitivity left out of the error test
  (``_SENSITIVITY_IN_ERROR_TEST = False``, CVODES ``errconS = false``);
- ``hold`` -- no step growth on the attempt accepted right after a failed one
  (``StepControl.hold_after_failure``);
- ``FIELD:VALUE`` -- a numeric ``StepControl`` field (``safety``, ``pi_beta``,
  ``maxfac``, ``minfac``, ``theta_target``).

Unset settings keep their shipped values. They are module settings of
``aquakin.integrate.forward_solve`` (and the ``StepControl`` the plant builds),
read when a solve is compiled: apply a spec before a plant's first solve.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import aquakin.integrate.forward_solve as forward_solve

_STEP_CONTROL = forward_solve.StepControl
_SHIPPED = (
    forward_solve._ANDERSON_DEPTH,
    forward_solve._STAGE_TEST_IN_VALUE_UNITS,
    forward_solve._SENSITIVITY_IN_ERROR_TEST,
)
_NUMERIC_FIELDS = ("safety", "pi_beta", "maxfac", "minfac", "theta_target")


@dataclass(frozen=True)
class LeanSettings:
    """One parsed spec (see the module docstring)."""

    spec: str
    anderson_depth: int
    value_units: bool
    sensitivity_in_error_test: bool
    control: tuple[tuple[str, object], ...]

    @classmethod
    def parse(cls, spec: str) -> LeanSettings:
        depth, value_units, sens_error = _SHIPPED
        control = {}
        for item in spec.split(","):
            kind, _, value = item.partition(":")
            if kind == "default":
                pass
            elif kind == "plain":
                depth = 0
            elif kind == "anderson":
                depth = int(value)
            elif kind == "value":
                value_units = True
            elif kind == "noerrcon":
                sens_error = False
            elif kind == "hold":
                control["hold_after_failure"] = True
            elif kind in _NUMERIC_FIELDS:
                control[kind] = float(value)
            else:
                raise ValueError(f"unknown lean-loop setting {item!r}")
        return cls(spec, depth, value_units, sens_error, tuple(control.items()))

    @property
    def tag(self) -> str:
        """The spec as a file-name-safe tag."""
        return self.spec.replace(":", "").replace(",", "+")

    def apply(self) -> None:
        """Set the module settings (and the plant's ``StepControl``) to this spec."""
        forward_solve._ANDERSON_DEPTH = self.anderson_depth
        forward_solve._STAGE_TEST_IN_VALUE_UNITS = self.value_units
        forward_solve._SENSITIVITY_IN_ERROR_TEST = self.sensitivity_in_error_test
        forward_solve.StepControl = functools.partial(_STEP_CONTROL, **dict(self.control))
