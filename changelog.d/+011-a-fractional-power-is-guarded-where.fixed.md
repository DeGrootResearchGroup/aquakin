A fractional power is guarded where it is singular, so `[X] ** 0.5` and
`sqrt([X])` behave the same. `d/dx x**e` is infinite at zero whenever `e < 1`,
which broke the gradient of a half-order rate law written as a power while
leaving the forward solve looking correct. **No value changes anywhere** — the
guard moves the derivative at a zero base from infinite to 0, exactly as the
`sqrt` guard does, and only where the exponent is below one, so `x**1` keeps
its derivative of 1 and `x**2` is untouched. A negative base under a
fractional exponent still gives NaN, since it has no real value. Verified
bit-identical across every shipped model's rates.
