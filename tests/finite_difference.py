"""Finite-difference references for gradient tests (``.claude/rules/tests.md``)."""


def central_fd(loss, x, i, rel_step=1e-4):
    """Central difference of the scalar ``loss`` with respect to ``x[i]``.

    The step is ``rel_step * x[i]``, so ``x[i]`` must be nonzero; ``x`` is a JAX
    array (it is perturbed with ``.at[i].add``).
    """
    h = rel_step * float(x[i])
    return (float(loss(x.at[i].add(h))) - float(loss(x.at[i].add(-h)))) / (2.0 * h)
