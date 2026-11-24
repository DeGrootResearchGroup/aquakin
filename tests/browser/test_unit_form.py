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


_SBR = """  sbr:
    kind: sbr
    model: asm1
    conditions: {T: 293.15}
    full_volume: 1000.0
    feed_flow: 5000.0
    decant_flow: 5000.0
    phases:
      - {name: fill, duration: 0.1, feed: true}
      - {name: react, duration: 0.3, kla: 240.0}
      - {name: decant, duration: 0.1, decant: true}
    settling: {kind: interface, v_settle: 400.0, area: 200.0}
"""


def _open_with_an_sbr(page, tmp_path):
    """Open a copy of BSM1 that also holds an SBR of three phases."""
    spec = tmp_path / "bsm1" / "flowsheet.yaml"
    with_sbr = spec.with_name("with_sbr.yaml")
    with_sbr.write_text(spec.read_text().replace("\nunits:\n", "\nunits:\n" + _SBR))
    page.get_by_placeholder("path/to/flowsheet.yaml").fill(str(with_sbr))
    page.get_by_role("button", name="Open", exact=True).click()
    expect(page.get_by_text(re.compile(r"^11 units · 11 connections"))).to_be_visible()


def test_phases_are_reordered_added_and_removed_in_the_table_and_reach_python_on_apply(
    page, tmp_path
):
    _open_with_an_sbr(page, tmp_path)
    page.locator('.vue-flow__node[data-id="sbr"]').click()
    cards = page.locator(".ak-records .v-sheet")
    expect(cards).to_have_count(3)
    # React before fill: its up arrow; the first card's up arrow is disabled.
    expect(cards.nth(0).locator("button:has(.mdi-arrow-up)")).to_be_disabled()
    cards.nth(1).locator("button:has(.mdi-arrow-up)").click()
    expect(cards.nth(0).get_by_label("name *")).to_have_value("react")
    # A settle phase added at the end and moved before the decant.
    page.get_by_role("button", name="Add a phase").click()
    expect(cards).to_have_count(4)
    cards.nth(3).get_by_label("name *").fill("settle")
    cards.nth(3).get_by_label("duration *").fill("0.05")
    cards.nth(3).get_by_label("settle", exact=True).check()
    cards.nth(3).locator("button:has(.mdi-arrow-up)").click()
    expect(cards.nth(2).get_by_label("name *")).to_have_value("settle")
    expect(cards.nth(2).get_by_label("settle", exact=True)).to_be_checked()
    page.get_by_role("button", name="Apply changes").click()
    page.get_by_role("tab", name="Spec").click()
    spec = page.locator("#app pre")
    order = r"name: react[\s\S]*name: fill[\s\S]*name: settle\n\s+duration: 0\.05[\s\S]*name: decant"
    expect(spec).to_contain_text(re.compile(order))

    # Removing the fill phase and applying again leaves the other three.
    page.get_by_role("tab", name="Unit").click()
    cards.nth(1).locator("button:has(.mdi-close)").click()
    expect(cards).to_have_count(3)
    page.get_by_role("button", name="Apply changes").click()
    page.get_by_role("tab", name="Spec").click()
    # Waits until Python's phases run react, settle, decant with no fill among them.
    no_fill = r"(?:(?!name: fill)[\s\S])*?"
    expect(spec).to_contain_text(
        re.compile(rf"phases:{no_fill}name: react{no_fill}name: settle{no_fill}name: decant")
    )


def test_choosing_a_settling_kind_draws_its_arguments(page, tmp_path):
    _open_with_an_sbr(page, tmp_path)
    page.locator('.vue-flow__node[data-id="sbr"]').click()
    expect(page.get_by_label("settling v_settle")).to_have_value("400")
    _choose(page, page.get_by_label("settling *"), "layered")
    # Python redraws the form with the layered model's arguments, keeping those
    # both kinds take.
    expect(page.get_by_label("settling n_layers")).to_be_visible()
    expect(page.get_by_label("settling v_settle")).to_have_value("400")
