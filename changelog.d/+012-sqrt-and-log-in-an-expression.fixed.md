`sqrt` and `log` in an expression of time are now the guarded definitions the
rate language uses (`aquakin.core.kernels`), so a solve driven by a pattern
such as `Q="2000 + 500*sqrt(t)"` differentiates. Unguarded they broke only the
gradient — the forward solve was unaffected — and did so as an exhausted
solver step budget rather than a visible NaN. This matters on a clock in
particular, since every solve starts at `t = 0`, which is where their
derivative is infinite. `log`'s guard protects the gradient only: its value at
zero is still `-inf`, so write `log(1 + t)` rather than `log(t)`. `log10` is
guarded the same way.
