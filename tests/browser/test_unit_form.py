"""The unit form's repeated rows (a reagent's composition), observed in the page.

The entries are added, chosen and removed in the browser, inside the form's
rows (nested state the page does not redraw on its own), and reach Python only
on Apply; what Python holds is read from the Spec tab.
"""

import re

import pytest

expect = pytest.importorskip("playwright.sync_api").expect


def _open_with_a_dosing_unit(page, tmp_path):
    """Open a copy of BSM1 that also holds a dosing unit with no reagent yet."""
    spec = tmp_path / "bsm1" / "flowsheet.yaml"
    text = spec.read_text()
    assert "\nunits:\n" in text
    dosed = spec.with_name("dosed.yaml")
    dosed.write_text(text.replace("\nunits:\n", "\nunits:\n  dose:\n    kind: dosing\n    flow: 2.0\n"))
    page.get_by_placeholder("path/to/flowsheet.yaml").fill(str(dosed))
    page.get_by_role("button", name="Open", exact=True).click()
    expect(page.get_by_text(re.compile(r"^11 units · 11 connections"))).to_be_visible()


def _choose(page, field, option):
    """Open a select (by its field: its input lies under the shown value) and
    click one of its options."""
    field.locator("xpath=ancestor::div[contains(@class, 'v-field')][1]").click()
    page.locator(".v-overlay--active .v-list-item-title").get_by_text(option, exact=True).click()


def test_a_reagent_is_composed_in_the_form_and_reaches_python_on_apply(page, tmp_path):
    _open_with_a_dosing_unit(page, tmp_path)
    page.locator('.vue-flow__node[data-id="dose"]').click()
    _choose(page, page.get_by_label("reagent model *"), "asm1")
    add = page.get_by_role("button", name="Add a species")
    add.click()
    add.click()
    species = page.get_by_label("species", exact=True)
    expect(species).to_have_count(2)  # both new entries are drawn
    _choose(page, species.nth(0), "SS")
    _choose(page, species.nth(1), "SNO")
    concentration = page.get_by_label("concentration", exact=True)
    concentration.nth(0).fill("250000")
    concentration.nth(1).fill("9")
    suffixes = page.locator(".ak-composition .v-text-field__suffix")
    expect(suffixes).to_have_text(["g_COD/m³", "g_N/m³"])  # each species' own unit
    page.get_by_role("button", name="Apply changes").click()
    page.get_by_role("tab", name="Spec").click()
    spec = page.locator("#app pre")
    expect(spec).to_contain_text("SS: 250000.0")
    expect(spec).to_contain_text("SNO: 9.0")

    # Removing an entry and applying again leaves the other.
    page.get_by_role("tab", name="Unit").click()
    page.locator(".ak-composition .mdi-close").nth(1).click()
    expect(page.get_by_label("species", exact=True)).to_have_count(1)
    page.get_by_role("button", name="Apply changes").click()
    page.get_by_role("tab", name="Spec").click()
    # Waits until Python's spec has a composition of SS alone.
    expect(spec).to_contain_text(re.compile(r"composition:\n\s+SS: 250000\.0\n(?!\s+SNO)"))
