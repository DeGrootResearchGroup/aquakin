"""The ozone-demand worked example runs and supports what its docs page says.

``docs/worked_example_design.md`` includes the code and the printed output of
``examples/ozone_demand_design.py``; this test runs the script and checks the
claims the page's text makes about that output, so the page cannot silently
drift.
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

    out = ex.main(output_dir=tmp_path)

    # Step 1: k1 alone, and the k2/[M]0 product, are flagged; nothing is identifiable.
    rep = out["step1"]
    assert len(rep.unidentifiable) == 2
    loads = rep.unidentifiable[1].loadings
    assert abs(loads["slow.k2"]) == pytest.approx(abs(loads["ic[0].M"]), abs=0.05)
    assert np.sign(loads["slow.k2"]) != np.sign(loads["ic[0].M"])
    assert not any(np.isfinite(v) for v in rep.relative_std.values())

    # Step 2: late samples do not help; early samples + UV254 make all identifiable.
    rows = out["step2"]
    assert not np.isfinite(rows["+ late samples"]["slow.k2"])
    assert not np.isfinite(rows["+ UV254"]["fast.k1"])
    assert all(np.isfinite(v) for v in rows["+ early samples + UV254"].values())

    # Step 3: the design improves, clusters early, and uses the full dose.
    opt = out["step3"]
    assert opt.success and opt.value > opt.initial_value
    t = np.asarray(opt.experiments[0].t_obs)
    assert np.sum(t < 30.0) >= 2
    assert float(opt.experiments[0].C0[0]) == pytest.approx(150.0)

    # Step 4: estimates within 3 sd of the truth; the Laplace sd matches the
    # design-stage prediction; the profile interval contains the truth.
    cal = out["step4"]
    for name, true in [("fast.k1", 0.01), ("slow.k2", 2e-6)]:
        est, sd = cal.params_named[name], cal.params_named_std[name]
        assert abs(est - true) < 3 * sd
        assert sd / est == pytest.approx(out["rel_designed"][name], rel=0.1)
    lo, hi = out["profile"].ci
    assert lo < 2e-6 < hi

    text = (tmp_path / "ozone_demand_output.txt").read_text()
    for marker in ["== Step 1 ==", "== Step 5 ==", "Profile likelihood:", "== End =="]:
        assert marker in text
    assert (tmp_path / "ozone_demand_design.png").exists()
    assert (tmp_path / "ozone_demand_profile.png").exists()
