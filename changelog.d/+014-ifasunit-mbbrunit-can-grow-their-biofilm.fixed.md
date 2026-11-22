`IFASUnit` / `MBBRUnit` can grow their biofilm. `dynamic_thickness=True` makes
the thickness `L_f` a state instead of a fixed parameter: with
`biofilm_density` setting the incompressible film's density and
`detachment_rate` / `detachment_law` closing the surface balance, the film
thickens as it grows,
thins when decay outruns growth, and sorts its species by depth. This is the
one-dimensional multispecies biofilm of Wanner & Gujer (1986): net production
anywhere inside an incompressible film has to displace material outward, and
that displacement carries the species with it. `L_f` is the final state entry
and is named `L_f` in `state_names()`. Validated against the closed form a
fully penetrated film must satisfy, `L_f = mu_net / lambda`, to better than
0.3%.
