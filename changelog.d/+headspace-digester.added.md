`HeadspaceDigesterUnit` is an anaerobic digester whose gas headspace belongs to
the unit rather than the model, so a model that declares its dissolved gases in
a `gases:` block can run in a digester unchanged. It follows BSM2's gas-liquid
transfer and biogas outflow, and `plant.mass_balance()` counts the biogas it
vents.
