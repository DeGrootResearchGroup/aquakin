`detachment_law="balance"` on a growing `IFASUnit` / `MBBRUnit`: **a specified
total thickness.** Detachment is set to exactly the velocity arriving at the
surface, so `dL_f/dt` is zero by construction and the film stays at
`biofilm_thickness` -- while the species still grow, advect and stratify, and
biomass still leaves at the surface. It is what a problem means by a specified
average thickness "maintained by detachment", and it takes no
`detachment_rate`. The rate-driven laws (`"quadratic"`, `"linear"`,
`"constant"`) are unchanged and still let the thickness follow from the
physics, so a thickness can now be either specified or predicted.
