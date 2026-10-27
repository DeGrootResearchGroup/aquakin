"""The ozone-demand worked example runs and supports what its docs page says.

``docs/worked_example_design.md`` quotes the output of
``examples/ozone_demand_design.py``; this test runs the script and checks the
qualitative claims the page makes, so the page cannot silently drift.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


@pytest.mark.slow
def test_ozone_demand_worked_example(tmp_path):
    sys.path.insert(0, str(EXAMPLES))
    try:
        import ozone_demand_design as ex
    finally:
        sys.path.remove(str(EXAMPLES))

    out = ex.main(figure_dir=tmp_path)

    # Step 1: k1 alone, and the k2/[M]0 product, are flagged.
    flagged = out["step1"]["unidentifiable"]
    assert len(flagged) == 2
    loads = flagged[1].loadings
    assert abs(loads["slow.k2"]) == pytest.approx(abs(loads["ic[0].M"]), abs=0.05)
    assert np.sign(loads["slow.k2"]) != np.sign(loads["ic[0].M"])

    # Step 2: only early samples plus UV254 make everything identifiable.
    rows = out["step2"]
    assert all(np.isfinite(v) for v in rows["+ early samples and UV254"]["relative_std"].values())
    assert not np.isfinite(rows["+ 5 late samples, 700-1800 s"]["relative_std"]["slow.k2"])

    # Step 3: optimisation improves the design and uses the full dose range.
    res = out["step3"]["result"]
    assert res.success and res.value > res.initial_value
    assert float(res.experiments[0].C0[0]) == pytest.approx(150.0)

    # Step 4: estimates within 3 sd of the truth, and the Laplace sd matches the
    # design-stage prediction.
    cal = out["step4"]["result"]
    pred = out["step3"]["relative_std_opt"]
    for name, true in [("fast.k1", 0.01), ("slow.k2", 2e-6)]:
        est, sd = cal.params_named[name], cal.params_named_std[name]
        assert abs(est - true) < 3 * sd
        assert sd / est == pytest.approx(pred[name], rel=0.1)
    lo, hi = out["step4"]["profile"].ci
    assert lo < 2e-6 < hi

    assert (tmp_path / "ozone_demand_design.png").exists()
    assert (tmp_path / "ozone_demand_profile.png").exists()
