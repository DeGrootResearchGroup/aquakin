`Plant.steady_state`'s pseudo-arclength fallback (and
`arclength_continuation_solve`) no longer reports a steady state on a
different branch as the operating point past a fold. Near a saddle-node the
continuation step could converge onto the other branch, so a target past the
fold came back as `method="arclength"`, `operating_point_exists=True` with
that branch's state instead of `"past_fold"`. A step that lands far from its
predictor is now rejected and retried shorter (`max_correction`, default
0.5 of the step), so the fold is detected. Screens that use
`operating_point_exists` to exclude samples (`steady_state_dgsm`) are
affected.
