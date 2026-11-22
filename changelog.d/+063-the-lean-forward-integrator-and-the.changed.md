The lean forward integrator and the staggered `Plant.solve_sensitivity` no
longer grow the step right after a rejected step, and the Newton iteration of
each implicit stage now stops on the change in the stage value rather than in
its slope, so its tolerance no longer depends on the time unit or the step
size; a stage whose first correction is already small enough is accepted
without a second. On a dynamic BSM2 window a forward solve used about a fifth
fewer right-hand-side evaluations, with the same accuracy against a
tight-tolerance solve, and a staggered `solve_sensitivity` started at a steady
state no longer stalls at tiny steps. Results change within the solver
tolerance.
