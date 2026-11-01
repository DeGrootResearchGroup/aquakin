"""Probe the step attempts of the lean loop on the BSM2 window.

Wraps ``forward_solve._integrate`` so every attempted step reports its start time
``t``, step size ``h``, error norm, largest Newton contraction rate and Newton
convergence, and ``forward_solve._wrms`` -- which the loop uses only for the
state's error norm -- so an error-test rejection can be split into one driven by
the state and one driven by the sensitivity (the step's error is the larger of
the two). The records are saved to ``--out`` (``.npz``). On the host it then
reports, per outcome (accepted / Newton non-converged / rejected on the state
error / rejected on the sensitivity error):

- how many attempts straddle an influent sample time (the synthesised influent
  is piecewise linear between 15-minute samples, so its derivative jumps there),
  against the share expected if the steps fell at random;
- the step size relative to the influent sample spacing;

and, for each error-test rejection that follows an accepted step:

- the step growth into it, and whether that accepted step itself followed a
  rejection (the reject -> accept -> grow -> reject cycle);
- the error jump ``en / (en_prev * (h / h_prev)**3)``: ``1`` when the error
  scales as the method's local error ``O(h^3)`` predicts, so a rejection with a
  jump near ``1`` is the controller overshooting, and one with a large jump is
  the error estimate itself changing (a kink in the right-hand side crossed).

Usage (the steady state is cached next to ``--out``)::

    python scripts/probe_step_rejections.py --days 0.25 --k 0
    python scripts/probe_step_rejections.py --days 0.25 --k 16
"""

from __future__ import annotations

import argparse
import collections
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from _bsm2_window import ATOL, RTOL, bsm2_window

import aquakin.integrate.forward_solve as forward_solve

_OUTCOMES = ("accepted", "nonconverged", "rejected_state", "rejected_sensitivity")


class AttemptRecorder:
    """Collect ``(t, h, en, rate, conv)`` per attempt and the state error norm."""

    def __init__(self):
        self.attempts = []
        self.state_errors = []

    def on_attempt(self, t, h, en, rate, conv):
        self.attempts.append((float(t), float(h), float(en), float(rate), bool(conv)))

    def on_state_error(self, en_y):
        self.state_errors.append(float(en_y))

    def table(self):
        if len(self.state_errors) != len(self.attempts):
            raise RuntimeError("state-error and attempt records are out of step")
        a = np.array(self.attempts, dtype=float).reshape(-1, 5)
        return {
            "t": a[:, 0],
            "h": a[:, 1],
            "en": a[:, 2],
            "rate": a[:, 3],
            "conv": a[:, 4].astype(bool),
            "en_y": np.array(self.state_errors),
        }


def install(recorder):
    integrate, wrms = forward_solve._integrate, forward_solve._wrms

    def recorded_integrate(attempt, z0, t0, t1, t_eval, control):
        def recorded_attempt(t, z, h):
            a = attempt(t, z, h)
            jax.debug.callback(recorder.on_attempt, t, h, a.en, a.rate, a.conv, ordered=True)
            return a

        return integrate(recorded_attempt, z0, t0, t1, t_eval, control)

    def recorded_wrms(v, scale):
        en = wrms(v, scale)
        jax.debug.callback(recorder.on_state_error, en, ordered=True)
        return en

    forward_solve._integrate = recorded_integrate
    forward_solve._wrms = recorded_wrms


def outcomes(rec):
    """The outcome of each attempt, one of ``_OUTCOMES``."""
    out = np.where(rec["en"] <= 1.0, "accepted", "rejected_sensitivity").astype(object)
    out[(rec["en"] > 1.0) & (rec["en_y"] > 1.0)] = "rejected_state"
    out[~rec["conv"]] = "nonconverged"
    return out


def report(rec, breakpoints):
    t, h, en, rate = rec["t"], rec["h"], rec["en"], rec["rate"]
    outcome = outcomes(rec)
    spacing = float(np.median(np.diff(breakpoints)))
    nxt = np.searchsorted(breakpoints, t, side="right")
    straddles = (nxt < len(breakpoints)) & (
        breakpoints[np.minimum(nxt, len(breakpoints) - 1)] < t + h * (1 - 1e-12)
    )
    print(f"influent sample spacing {spacing:.6f} d; {len(t)} attempts")
    for cls in _OUTCOMES:
        m = outcome == cls
        if not m.any():
            print(f"== {cls}: 0")
            continue
        r = h[m] / spacing
        print(
            f"== {cls}: {m.sum()}  straddle a sample time {straddles[m].mean():.0%} "
            f"(random placement {np.minimum(r, 1.0).mean():.0%})  h/spacing median "
            f"{np.median(r):.2f} p90 {np.percentile(r, 90):.2f}  Newton rate median "
            f"{np.median(rate[m]):.2f} p90 {np.percentile(rate[m], 90):.2f}"
        )

    rejected = np.isin(outcome, ("rejected_state", "rejected_sensitivity"))
    preceding = collections.Counter()
    growth, jump, en_acc, cycle = [], [], [], []
    for i in np.flatnonzero(rejected):
        if i == 0:
            continue
        if outcome[i - 1] != "accepted":
            preceding[f"after {outcome[i - 1]}"] += 1
            continue
        ratio = h[i] / h[i - 1]
        preceding["after an accepted step" + (" (grew)" if ratio > 1.0 else "")] += 1
        growth.append(ratio)
        en_acc.append(en[i - 1])
        jump.append(en[i] / (max(en[i - 1], 1e-30) * ratio**3))
        cycle.append(i >= 2 and outcome[i - 2] != "accepted")
    print("== what preceded a rejection:", dict(preceding))
    if growth:
        g, j = np.array(growth), np.array(jump)
        print(
            f"   growth into it: median {np.median(g):.2f} p90 {np.percentile(g, 90):.2f}, "
            f"at the 2x cap {np.mean(g > 1.99):.0%}; the accepted step itself followed a "
            f"failed attempt {np.mean(cycle):.0%}"
        )
        print(
            f"   accepted step's en: median {np.median(en_acc):.2f}; error jump "
            f"en/(en_prev*(h/h_prev)^3): median {np.median(j):.1f} p25 "
            f"{np.percentile(j, 25):.1f} p75 {np.percentile(j, 75):.1f}; <2: {np.mean(j < 2):.0%}, "
            f">10: {np.mean(j > 10):.0%}"
        )
    # The same jump measured over consecutive accepted steps, for comparison:
    # how well the O(h^3) model predicts the next error when nothing went wrong.
    acc = outcome == "accepted"
    pair = acc[:-1] & acc[1:]
    if pair.any():
        jj = en[1:][pair] / (np.maximum(en[:-1][pair], 1e-30) * (h[1:][pair] / h[:-1][pair]) ** 3)
        print(
            f"   accepted -> accepted error jump: median {np.median(jj):.1f} p25 "
            f"{np.percentile(jj, 25):.1f} p75 {np.percentile(jj, 75):.1f}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=float, default=0.25)
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--anderson-depth", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("probe_step_rejections.npz"))
    args = ap.parse_args()
    if args.anderson_depth is not None:
        forward_solve._ANDERSON_DEPTH = args.anderson_depth

    plant, params, y_ss, asm_params = bsm2_window(args.out.with_name("bsm2_open_loop_ss.npy"))
    recorder = AttemptRecorder()
    install(recorder)
    start = time.perf_counter()
    *_, stats = plant.solve_sensitivity(
        params,
        asm_params[: args.k],
        t_span=(0.0, args.days),
        t_eval=jnp.array([0.0, args.days]),
        y0=y_ss,
        rtol=RTOL,
        atol=ATOL,
        max_steps=200_000,
        return_stats=True,
    )
    print(
        f"days={args.days} k={args.k} anderson_depth={forward_solve._ANDERSON_DEPTH} "
        f"solve {time.perf_counter() - start:.0f}s "
        f"attempted/accepted/nonconverged/error-rejected {[int(v) for v in stats]}",
        flush=True,
    )
    rec = recorder.table()
    np.savez(args.out, **rec)
    breakpoints = np.unique(np.concatenate([np.asarray(inf.t) for inf in plant.influents.values()]))
    report(rec, breakpoints)


if __name__ == "__main__":
    main()
