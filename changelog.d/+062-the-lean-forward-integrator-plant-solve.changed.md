The lean forward integrator (`Plant.solve(forward_fast=True)`) and the
staggered `Plant.solve_sensitivity` now Anderson-accelerate the Newton
iteration of each implicit stage, using the step's Jacobian factorization as
the preconditioner. Their results change within the solver tolerance. On a
dynamic BSM2 window, half as many steps failed Newton convergence and a
forward solve ran about 30% faster.
