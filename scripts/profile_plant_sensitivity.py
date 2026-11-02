"""Profile ``Plant.solve_sensitivity`` on a BSM2 dynamic window (issue #654).

Runs the window of ``_bsm2_window.py`` (open-loop BSM2 from its constant-influent
steady state, synthesised dry-weather influent, ``rtol=1e-4`` / ``atol=1e-3``)
and reports, per sensitivity-parameter count ``k`` (the first ``k`` ASM1
parameters):

- the wall time of a first call (compile + run) and of later calls (run only,
  the compiled solve reused from the plant's cache);
- the step counts: attempted, non-converged (Newton divergence / no
  convergence) and error-test rejections.

For the staggered corrector the counts come from the solve itself
(``return_stats=True``). For the simultaneous corrector they are counted by
wrapping ``diffrax.Kvaerno3.step`` / ``PIDController.adapt_step_size`` with a
``jax.debug.callback`` (result codes 7/8 are the root finder's max-steps /
divergence failures; every other rejection is an error-test failure).

``--variants`` (staggered corrector only) compares settings of the lean loop,
each a spec of ``_lean_settings.py`` (e.g. ``default``, ``plain``,
``anderson:2``, ``kappa:0.03,noerrcon``). They are read when a solve is compiled,
so each variant gets its own plant, compiled once, and the cached calls of all
variants are **interleaved** (A, B, A, B, ...): wall times on a shared machine
drift between calls (by up to 2x here), and interleaving puts every variant under
the same drift. Compare the fastest call of each.

The state ``ys`` and sensitivity ``S`` of each run are saved next to ``--out``.
With ``--reference-rtol`` each first call also reports its error against a
tight-tolerance solve under the default settings (computed once per ``k`` and
cached next to ``--out``): ``err_y`` per state component relative to its peak
(floored at ``atol``), ``err_S`` per sensitivity column relative to its peak,
the largest over the save times.

Usage::

    python scripts/profile_plant_sensitivity.py --corrector staggered --k 0 1 16
    python scripts/profile_plant_sensitivity.py --corrector simultaneous --k 0 16
    python scripts/profile_plant_sensitivity.py --days 1 --k 0 16 --calls 6 \
        --variants plain anderson:2
    python scripts/profile_plant_sensitivity.py --days 1 --k 0 --calls 6 \
        --variants default kappa:0.03 safety:0.8
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from _bsm2_window import ATOL, RTOL, bsm2_window
from _lean_settings import LeanSettings

_NONLINEAR_FAILURE_CODES = (7, 8)  # diffrax RESULTS: nonlinear max steps / divergence


class _DiffraxStepCounter:
    """Count Kvaerno3 step results and controller rejections of diffrax solves."""

    def __init__(self):
        import diffrax

        self.attempted = self.nonconverged = self.rejected = 0
        step, adapt = diffrax.Kvaerno3.step, diffrax.PIDController.adapt_step_size

        def counted_step(solver, *a, **kw):
            out = step(solver, *a, **kw)
            jax.debug.callback(self._on_step, out[-1]._value)
            return out

        def counted_adapt(controller, *a, **kw):
            out = adapt(controller, *a, **kw)
            jax.debug.callback(self._on_adapt, out[0])
            return out

        diffrax.Kvaerno3.step = counted_step
        diffrax.PIDController.adapt_step_size = counted_adapt

    def _on_step(self, code):
        self.attempted += 1
        self.nonconverged += int(int(code) in _NONLINEAR_FAILURE_CODES)

    def _on_adapt(self, keep):
        self.rejected += int(not bool(keep))

    def reset(self):
        self.attempted = self.nonconverged = self.rejected = 0

    def counts(self):
        return self.attempted, self.nonconverged, self.rejected - self.nonconverged


class _Variant:
    """One lean-loop settings spec (``_lean_settings.py``), with a plant compiled
    under it."""

    def __init__(self, spec, cache):
        self.settings = LeanSettings.parse(spec)
        self.tag = self.settings.tag
        self.plant, self.params, self.y_ss, self.asm_params = bsm2_window(cache)

    def configure(self):
        """Apply the module settings; return the ``solve_sensitivity`` keywords."""
        self.settings.apply()
        return self.settings.solve_kwargs


def _solve(
    plant, params, y_ss, wrt, args, t_eval, *, rtol=RTOL, atol=ATOL, max_steps=200_000, **extra
):
    start = time.perf_counter()
    out = plant.solve_sensitivity(
        params,
        wrt,
        t_span=(0.0, args.days),
        t_eval=t_eval,
        y0=y_ss,
        rtol=rtol,
        atol=atol,
        factormax=args.factormax,
        max_steps=max_steps,
        corrector=args.corrector,
        return_stats=True,
        **extra,
    )
    jax.block_until_ready(out[2])
    return out, time.perf_counter() - start


def _reference(k, args, cache, t_eval):
    """A tight-tolerance solve under the default settings (cached on disk).

    The lean loop at ``rtol = --reference-rtol`` with ``atol`` scaled by the
    same factor, so each variant's error against it is its own error, not the
    disagreement of two solves at the measured tolerance.
    """
    path = args.out.with_name(f"reference_k{k}_days{args.days}_rtol{args.reference_rtol:g}.npz")
    if path.exists():
        ref = np.load(path)
        return ref["y"], ref["S"]
    variant = _Variant("default", cache)
    variant.configure()
    factor = args.reference_rtol / RTOL
    out, wall = _solve(
        variant.plant,
        variant.params,
        variant.y_ss,
        variant.asm_params[:k],
        args,
        t_eval,
        rtol=args.reference_rtol,
        atol=ATOL * factor,
        max_steps=2_000_000,
    )
    print(
        f"reference k={k} rtol={args.reference_rtol:g} wall={wall:.0f}s "
        f"steps={[int(v) for v in out[3]]}",
        flush=True,
    )
    np.savez(path, y=np.asarray(out[1]), S=np.asarray(out[2]))
    variant.plant._solve_cache.invalidate()
    jax.clear_caches()
    return np.asarray(out[1]), np.asarray(out[2])


def _errors(out, ref):
    """Largest error over the save times: per state component relative to its
    peak (floored at ``ATOL``), per sensitivity column relative to its peak."""
    y, S = np.asarray(out[1]), np.asarray(out[2])
    y_ref, S_ref = ref
    peak_y = np.maximum(np.max(np.abs(y_ref), axis=0), ATOL)
    err_y = float(np.max(np.max(np.abs(y - y_ref), axis=0) / peak_y))
    if S.shape[-1] == 0:
        return err_y, float("nan")
    peak_S = np.maximum(np.max(np.abs(S_ref), axis=(0, 1)), 1e-300)
    err_S = float(np.max(np.max(np.abs(S - S_ref), axis=(0, 1)) / peak_S))
    return err_y, err_S


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corrector", choices=("staggered", "simultaneous"), default="staggered")
    ap.add_argument("--k", type=int, nargs="+", default=[0, 1, 16])
    ap.add_argument("--days", type=float, default=5.0)
    ap.add_argument("--factormax", type=float, default=None)
    ap.add_argument("--calls", type=int, default=2, help="calls per k (the first compiles)")
    ap.add_argument("--variants", nargs="+", default=None, help="stage-iteration settings")
    ap.add_argument(
        "--reference-rtol",
        type=float,
        default=None,
        help="also report each variant's error against a solve at this rtol",
    )
    ap.add_argument("--out", type=Path, default=Path("profile_plant_sensitivity.txt"))
    args = ap.parse_args()
    if args.variants and args.corrector != "staggered":
        ap.error("--variants applies to the staggered corrector only")

    cache = args.out.with_name("bsm2_open_loop_ss.npy")
    variants = [_Variant(spec, cache) for spec in args.variants] if args.variants else None
    if variants is None:
        plant, params, y_ss, asm_params = bsm2_window(cache)
        runs = [(args.corrector, plant, params, y_ss, asm_params, dict)]
    else:
        runs = [
            (f"{args.corrector}_{v.tag}", v.plant, v.params, v.y_ss, v.asm_params, v.configure)
            for v in variants
        ]
    counter = _DiffraxStepCounter() if args.corrector == "simultaneous" else None
    t_eval = jnp.linspace(0.0, args.days, int(args.days) + 1)

    with args.out.open("a") as log:
        for k in args.k:
            ref = _reference(k, args, cache, t_eval) if args.reference_rtol else None
            for call in range(1, args.calls + 1):
                for tag, plant, params, y_ss, asm_params, configure in runs:
                    # The module settings are read only when this plant first
                    # compiles (call 1); the keywords go to every call.
                    extra = configure()
                    if counter is not None:
                        counter.reset()
                    out, wall = _solve(plant, params, y_ss, asm_params[:k], args, t_eval, **extra)
                    if counter is not None:
                        attempted, nonconverged, error_rejected = counter.counts()
                    else:
                        stats = out[3]
                        attempted = int(stats.attempted)
                        nonconverged = int(stats.nonconverged)
                        error_rejected = int(stats.error_rejected)
                    line = (
                        f"{tag} days={args.days} factormax={args.factormax} k={k} "
                        f"call={call} wall={wall:.1f}s attempted={attempted} "
                        f"nonconverged={nonconverged} error_rejected={error_rejected} "
                        f"finite={bool(np.all(np.isfinite(np.asarray(out[2]))))}"
                    )
                    if ref is not None and call == 1:
                        err_y, err_S = _errors(out, ref)
                        line += f" err_y={err_y:.2e} err_S={err_S:.2e}"
                    print(line, flush=True)
                    log.write(line + "\n")
                    log.flush()
                    if call == 1:
                        np.save(args.out.with_name(f"{tag}_k{k}_S.npy"), np.asarray(out[2]))
                        np.save(args.out.with_name(f"{tag}_k{k}_y.npy"), np.asarray(out[1]))
            # Several whole-plant programs per k are live at once; drop them before
            # compiling the next k, or the process runs out of memory.
            for _, plant, *_ in runs:
                plant._solve_cache.invalidate()
            jax.clear_caches()


if __name__ == "__main__":
    main()
