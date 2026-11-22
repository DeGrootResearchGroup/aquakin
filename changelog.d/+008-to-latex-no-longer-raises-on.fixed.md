`to_latex` no longer raises on a node type it has no explicit branch for; it
renders any rate function as `\mathrm{name}(args)`. Previously a model using
such a node could be solved but not written up.
