`IFASUnit.layer_resolution(state)` reports, per soluble species, the largest
fractional concentration drop across one biofilm layer -- how well `n_layers`
resolves the gradient the substrate flux is read from. A few per cent is well
resolved; a value near 1 means the substrate is consumed inside a single layer
and `n_layers` is too low. It distinguishes a limiting substrate from one that
penetrates freely.
