"""The model panel in the page: a rate edited, its error placed, and the edit
saved as a new model of the project.

The draft is edited in the browser and sent when a field loses focus, so what
this pins is that the page's draft reaches Python, Python's answers come back
beside the text without redrawing it, and Apply's dialog writes the files.
"""

import re

import pytest
import yaml

sync_api = pytest.importorskip("playwright.sync_api")
expect = sync_api.expect


def test_a_rate_is_edited_its_error_shown_and_the_edit_saved_as_a_new_model(page, tmp_path):
    page.get_by_role("tab", name="Model").click()
    decay = page.locator(".ak-reaction").filter(has_text="decay_heterotrophs").first
    rate = decay.locator(".ak-rate input")
    expect(rate).to_have_value("rho_hetero_decay")

    rate.fill("bH * ([XB_H]")
    rate.press("Tab")  # the draft is sent as the field loses focus
    expect(decay.get_by_text(re.compile(r"at position 12"))).to_be_visible()
    expect(decay.locator(".ak-pointer")).to_have_text("bH * ([XB_H]\n            ^")
    expect(page.get_by_role("button", name="Apply", exact=True)).to_be_disabled()

    rate.fill("2 * bH * [XB_H]")
    rate.press("Tab")
    expect(page.get_by_text("aquakin loaded and compiled the edited model.")).to_be_visible()
    expect(decay.get_by_text("species XB_H · parameters bH")).to_be_visible()
    expect(rate).to_have_value("2 * bH * [XB_H]")  # the answers left the typing alone

    page.get_by_role("button", name="Apply", exact=True).click()
    dialog = page.get_by_role("dialog")
    expect(dialog.get_by_label("name of the new model")).to_have_value("asm1_bsm1")
    expect(dialog.locator(".ak-dialog-unit")).to_have_count(10)
    dialog.get_by_role("button", name="Save new model").click()
    expect(dialog).to_be_hidden()

    page.get_by_role("tab", name="Spec").click()
    expect(page.locator("#app pre")).to_contain_text("source: models/asm1_bsm1.yaml")
    model = yaml.safe_load((tmp_path / "bsm1" / "models" / "asm1_bsm1.yaml").read_text())
    assert model["model"] == {"name": "asm1_bsm1", "extends": "asm1"}
    assert model["reactions"] == [{"name": "decay_heterotrophs", "rate": "2 * bH * [XB_H]"}]
    spec = yaml.safe_load((tmp_path / "bsm1" / "flowsheet.yaml").read_text())
    assert {u["model"] for u in spec["units"].values()} == {"asm1_bsm1"}
