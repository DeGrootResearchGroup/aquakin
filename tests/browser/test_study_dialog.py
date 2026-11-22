"""The study dialog and the unit form, observed in the page.

The study draft is nested, and the page does not redraw on a nested change:
rows added or removed, and a stream chosen in a row, reach the page only
because each edit reassigns the draft. These tests fail when one of them does
not (the rule in ``.claude/rules/ui.md``).
"""

import re

import pytest
import yaml

from aquakin.study import plant_choices

expect = pytest.importorskip("playwright.sync_api").expect


def _new_study(page):
    page.get_by_role("tab", name="Studies").click()
    page.get_by_role("button", name="New study").click()
    dialog = page.locator(".v-dialog")
    expect(dialog.get_by_role("button", name="Create")).to_be_visible()
    return dialog


def _choose(page, field, option):
    """Open a select or combobox and click one of its options."""
    field.click()
    page.locator(".v-overlay--active .v-list-item-title").get_by_text(option, exact=True).click()


def _row(dialog, page, i):
    """The remove button of the ``i``-th measured-channel row."""
    rows = dialog.locator(".d-flex.ga-2.mb-1").filter(has=page.get_by_label("Stream", exact=True))
    return rows.nth(i).locator("button.v-btn")


def _options(page, field):
    """The options a select or combobox offers, as listed when it opens."""
    field.click()
    items = page.locator(".v-overlay--active .v-list-item-title")
    items.first.wait_for()
    texts = items.all_inner_texts()
    page.keyboard.press("Escape")
    return texts


def test_stream_rows_appear_go_and_offer_their_streams_channels(page, tmp_path):
    """Adding and removing a measured-channel row redraws the dialog; a row's
    channels are the ones aquakin offers for the stream chosen in it; and
    Create writes what the rows hold."""
    project = tmp_path / "bsm1"
    dialog = _new_study(page)
    dialog.get_by_label("Name").fill("measured")
    stream = dialog.get_by_label("Stream", exact=True)
    channels = dialog.get_by_label(re.compile(r"^Channels"))

    dialog.get_by_role("button", name="Add a stream").click()
    expect(stream).to_have_count(1)
    dialog.get_by_role("button", name="Add a stream").click()
    expect(stream).to_have_count(2)
    _row(dialog, page, 1).click()
    expect(stream).to_have_count(1)

    _choose(page, stream, "effluent")
    spec = yaml.safe_load((project / "flowsheet.yaml").read_text())
    offered = plant_choices(spec, base_dir=project)["channels"]["effluent"]
    assert _options(page, channels) == offered
    _choose(page, channels, "SNH")
    _choose(page, channels, "TSS")
    page.keyboard.press("Escape")

    dialog.get_by_role("button", name="Create").click()
    expect(dialog).to_have_count(0)
    written = yaml.safe_load((project / "studies" / "measured" / "study.yaml").read_text())
    assert written["study"]["observables"] == [{"stream": "effluent", "channels": ["SNH", "TSS"]}]


def test_a_new_row_starts_empty_after_another_was_removed(page):
    """Removing a row and adding one gives an empty row, not the removed
    row's values (the rows are keyed by position)."""
    dialog = _new_study(page)
    stream = dialog.get_by_label("Stream", exact=True)
    dialog.get_by_role("button", name="Add a stream").click()
    _choose(page, stream, "effluent")
    _row(dialog, page, 0).click()
    expect(stream).to_have_count(0)
    dialog.get_by_role("button", name="Add a stream").click()
    expect(stream).to_have_count(1)
    expect(stream).to_have_value("")


def test_a_fields_help_is_shown_in_the_fixed_panel_and_apply_does_not_move(page):
    """Focusing a field shows its help in the panel below the form; leaving the
    field (as pressing Apply does) does not move Apply, so the press and the
    release land on it and the typed value is applied."""
    page.locator('.vue-flow__node[data-id="tank1"]').click()
    apply = page.get_by_role("button", name="Apply")
    before = apply.bounding_box()
    volume = page.get_by_label("volume *")
    volume.click()
    panel = page.locator(".v-sheet.text-caption")
    expect(panel).to_have_text(re.compile(r"^volume \*\s*Tank liquid volume \(m³\)\.$"))
    assert apply.bounding_box() == before
    volume.fill("1250")
    apply.click()
    expect(page.locator('.vue-flow__node[data-id="tank1"]').locator(".ak-summary")).to_have_text(
        "volume 1250"
    )
    assert apply.bounding_box() == before
