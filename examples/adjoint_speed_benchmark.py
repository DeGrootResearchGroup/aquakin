"""Reverse-mode gradient speed: capped autodiff vs the cap-free discrete adjoint.

Two ways to get a reverse-mode parameter gradient through a stiff reaction-network
solve:

- ``jax_adjoint`` -- let JAX/diffrax differentiate the whole solve
  (``RecursiveCheckpointAdjoint``). For stiff networks this overflows above a
  step-size threshold, so the solver must carry a ``dtmax`` cap that forces many
  small steps over the whole integration.
- ``stable_adjoint`` -- the hand-written ESDIRK discrete adjoint
  (:func:`aquakin.esdirk_adjoint_solve`): a robust *adaptive* Kvaerno5 forward
  whose backward is an explicit per-step transposed-stage solve. No cap.

Both compute the same gradient (model derivatives are autodiff in both); they
differ in how the integrator's adjoint is formed, and therefore in step count.
This script times both, with matched forced-step forwards (so the gradients are
directly comparable), across a few integration spans.

What to expect: at a tight tolerance the capped step count grows roughly linearly
with the span (the cap, not accuracy, sets the step), while the adaptive Kvaerno5
step count stays nearly flat (almost all steps are in the initial stiff
transient; the smooth tail is cheap for an adaptive high-order method). So the two
are about par for short batch fits and the discrete adjoint pulls ahead for the
longer, multi-day runs -- on top of needing no cap to tune and never overflowing.

Run with::

    python examples/adjoint_speed_benchmark.py
"""

import time

import diffrax
import jax
import jax.numpy as jnp

import aquakin
from aquakin.integrate.discrete_adjoint import esdirk_adjoint_solve

RTOL, ATOL = 1e-7, 1e-10
# The loosest cap that keeps the capped autodiff gradient finite at this rtol on
# this network. (Too loose -> overflow; the value shifts with rtol/span/model --
# which is exactly the tuning the discrete adjoint removes.)
DTMAX = 3e-4


def _forced_controller(t_obs, dtmax):
    # Force the adaptive controller to land steps exactly on the observation
    # times, so both methods integrate the identical discrete solve.
    return diffrax.ClipStepSizeController(
        diffrax.PIDController(rtol=RTOL, atol=ATOL, dtmax=dtmax), step_ts=t_obs
    )


def _count_steps(rhs, C0, params, span, t_obs, dtmax):
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t, y, a: rhs(t, y, a)), diffrax.Kvaerno5(),
        0.0, span, 1e-6, C0, args=params,
        stepsize_controller=_forced_controller(t_obs, dtmax),
        saveat=diffrax.SaveAt(steps=True), max_steps=2_000_000,
    )
    return int(jnp.sum(jnp.isfinite(sol.ts)))


def _time_ms(fn, reps=3):
    fn().block_until_ready()  # compile once, not timed
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn().block_until_ready()
        times.append(time.perf_counter() - t0)
    return min(times) * 1e3


def main() -> None:
    net = aquakin.load_network("wats_sewer_khalil_paper_balanced")
    cond = net.default_conditions(1)
    C0 = net.default_concentrations()
    params = net.default_parameters()
    fields = cond.fields
    rhs = lambda t, y, p: net.dCdt(y, p, fields, 0)
    si = net.species_index["S_SO4"]

    print(f"stiff Khalil network: n={net.n_species} species, {net.n_params} params")
    print(f"rtol={RTOL}, atol={ATOL}; capped autodiff uses dtmax={DTMAX}\n")
    header = (f"{'span(d)':>7} {'capped':>8} {'adaptive':>9} "
              f"{'jax_adjoint':>12} {'stable_adjoint':>15} {'speedup':>8} {'rel.diff':>9}")
    print(header)
    print("-" * len(header))

    for span in (0.3, 1.0, 3.0):
        t_obs = jnp.linspace(span / 5, span, 5)
        n_capped = _count_steps(rhs, C0, params, span, t_obs, DTMAX)
        n_adaptive = _count_steps(rhs, C0, params, span, t_obs, None)
        max_steps = n_adaptive * 3 + 200

        def loss_jax(p, span=span, t_obs=t_obs):
            sol = diffrax.diffeqsolve(
                diffrax.ODETerm(lambda t, y, a: rhs(t, y, a)), diffrax.Kvaerno5(),
                0.0, span, 1e-6, C0, args=p,
                stepsize_controller=_forced_controller(t_obs, DTMAX),
                adjoint=diffrax.RecursiveCheckpointAdjoint(),
                saveat=diffrax.SaveAt(ts=t_obs), max_steps=2_000_000,
            )
            return jnp.sum(sol.ys[:, si] ** 2) + 1e-3 * jnp.sum(sol.ys ** 2)

        def loss_stable(p, span=span, t_obs=t_obs, max_steps=max_steps):
            ys = esdirk_adjoint_solve(rhs, C0, p, (0.0, span), t_obs,
                                      rtol=RTOL, atol=ATOL, max_steps=max_steps)
            return jnp.sum(ys[:, si] ** 2) + 1e-3 * jnp.sum(ys ** 2)

        g_jax = jax.jit(jax.grad(loss_jax))
        g_stable = jax.jit(jax.grad(loss_stable))
        t_jax = _time_ms(lambda: g_jax(params))
        t_stable = _time_ms(lambda: g_stable(params))
        rel = float(jnp.linalg.norm(g_jax(params) - g_stable(params))
                    / (jnp.linalg.norm(g_jax(params)) + 1e-30))
        print(f"{span:7.1f} {n_capped:8d} {n_adaptive:9d} {t_jax:10.0f}ms "
              f"{t_stable:13.0f}ms {t_jax / t_stable:7.2f}x {rel:9.1e}")


if __name__ == "__main__":
    main()
