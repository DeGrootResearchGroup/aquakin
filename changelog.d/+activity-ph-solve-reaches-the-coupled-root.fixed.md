`solve_ph` with `activity_model="davies"` or `"debye_huckel"` (the activity
corrections a `speciation:` block opts into, or
`load_model(..., activity_model=...)`) no longer returns a wrong pH for some
inputs. The ionic strength was updated in the middle of the bracketed pH
search, which could leave the true root outside the bracket. The search then
stopped at the bracket edge with no warning, at any iteration cap. In a sweep
of digester-like compositions about 12% of results were wrong: by up to
0.1 pH between pH 5 and 9, and by several pH units at the extremes. The pH is
now solved at a fixed ionic strength, and the ionic strength is updated
between solves until it settles. The ideal default (`activity_model="none"`)
and ADM1 with activities at its default state are unaffected.
