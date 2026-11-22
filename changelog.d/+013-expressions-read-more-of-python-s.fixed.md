Expressions read more of Python's syntax, in rate expressions and expressions
of time alike: a number may carry Python's underscore separators
(`18446_000`), a call may end with a trailing comma (`monod([SS], KS,)`), and
`max` and `min` take any number of arguments (`max(a, b, c)`) as Python's do.
Nothing that parsed before parses differently — every expression in every
shipped model is unchanged. What stays deliberately absent is anything that
cannot be traced: comparisons, `if`/`else`, `%` and `//`. `^` is still refused
rather than read as exclusive-or, since a reader means a power by it.
