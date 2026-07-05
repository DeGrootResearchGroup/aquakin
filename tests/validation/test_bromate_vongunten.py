"""Scientific validation against published bromate-formation data.

This test is intentionally a stub. To enable it:
1. Digitize ozone, bromide, and bromate trajectories from Acero & von Gunten
   (2001) figures (or equivalent published trajectories under known
   experimental conditions: pH, T, alkalinity, DOC).
2. Encode them as fixtures alongside the experimental conditions.
3. Solve the shipped ``ozone_bromate`` model against the same initial state
   and assert agreement within reasonable tolerances.

Reference: Acero, J.L. & von Gunten, U. (2001), J. AWWA 93(10), 90-100.
"""

import pytest


@pytest.mark.validation
@pytest.mark.skip(reason="TODO: digitize Acero & von Gunten (2001) trajectories")
def test_bromate_trajectory_matches_acero_2001():
    raise NotImplementedError
