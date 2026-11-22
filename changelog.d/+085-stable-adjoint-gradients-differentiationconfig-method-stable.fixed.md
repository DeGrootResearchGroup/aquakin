Stable-adjoint gradients (`DifferentiationConfig(method="stable")`) on large
plants such as BSM2 could be wrong while staying finite. The colored backward
Jacobian, on by default from 100 states, missed couplings: those from the
settler to the sludge train and the reactors, and the loops through the
solids-dependent thickener and dewatering flows. Gradients from the activated
sludge into the digester (for example `asm1.muA` on digester methane) were off
by up to 2.3×. The sparsity pattern now covers these couplings. The colored
backward is also checked against the dense Jacobian at several states,
including the solve's start state, and falls back to the dense Jacobian with
a warning if anything is missed. Gradients within one domain were unaffected
or only slightly affected. If you computed cross-domain BSM2 gradients with
the default settings, recompute them; `IntegratorConfig(colored_jacobian=False)`
gave correct values before this fix.
