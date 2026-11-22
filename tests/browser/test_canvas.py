"""The canvas's sync rules (``.claude/rules/ui.md``), observed in the page.

The header chip counts Python's document ("10 units · 11 connections"), so
what Python holds is read there, and what the canvas draws from its elements.
"""

import re

import pytest

expect = pytest.importorskip("playwright.sync_api").expect

NODE = ".vue-flow__node"
EDGE = ".vue-flow__edge"


def _counts(page, units, connections):
    expect(
        page.get_by_text(re.compile(rf"^{units} units · {connections} connections"))
    ).to_be_visible()


def _undo(page):
    return page.get_by_title("Undo the last edit of the flowsheet")


def _unit(page, uid):
    return page.locator(f'{NODE}[data-id="{uid}"]')


def test_every_loaded_wire_is_drawn(page):
    """Every connection Python sends is drawn (none is dropped by validation
    meant for wires being drawn)."""
    _counts(page, 10, 11)
    expect(page.locator(NODE)).to_have_count(10)
    expect(page.locator(EDGE)).to_have_count(11)


def _handle(page, uid, port):
    return _unit(page, uid).locator(f'.vue-flow__handle[data-handleid="{port}"]')


def _record_messages(page):
    """Keep every message the snackbar shows (it closes after a few seconds)."""
    page.evaluate(
        """() => {
          window.akMessages = [];
          new MutationObserver(() => {
            for (const el of document.querySelectorAll('.v-snackbar__content')) {
              const text = el.innerText.trim();
              if (text && window.akMessages.at(-1) !== text) window.akMessages.push(text);
            }
          }).observe(document.body, {childList: true, subtree: true, characterData: true});
        }"""
    )


def _messages(page):
    return page.evaluate("() => window.akMessages")


def _draw_wire(page, source, target):
    """Drag from one handle to another, as a user draws a wire."""
    a, b = source.bounding_box(), target.bounding_box()
    page.mouse.move(a["x"] + a["width"] / 2, a["y"] + a["height"] / 2)
    page.mouse.down()
    page.mouse.move(b["x"] + b["width"] / 2, b["y"] + b["height"] / 2, steps=12)
    page.mouse.up()


def test_a_wire_into_a_fed_input_is_refused_while_drawn(page):
    """The handles judge a wire being drawn: one into an input that is already
    fed never reaches Python (which would refuse it with a message), while a
    valid one, drawn after it, does."""
    _record_messages(page)
    _unit(page, "influent").click()
    page.keyboard.press("Delete")  # frees inlet_mix.fresh
    _counts(page, 9, 10)
    _draw_wire(page, _handle(page, "clarifier", "overflow"), _handle(page, "tank2", "inlet"))
    _draw_wire(page, _handle(page, "clarifier", "overflow"), _handle(page, "inlet_mix", "fresh"))
    _counts(page, 9, 11)  # Python has handled both, in order
    assert _messages(page) == ["removed influent: Undo brings it back"]


def test_pythons_updates_are_applied_as_a_diff(page):
    """A selection and a renamed unit come back from Python as new lists; the
    canvas updates its nodes in place rather than rebuilding them."""
    page.eval_on_selector_all(NODE, "els => els.forEach(e => e.dataset.kept = 'yes')")
    _unit(page, "tank1").click()
    label = page.get_by_label("label", exact=True)
    label.fill("First tank")
    page.get_by_role("button", name="Apply").click()
    expect(_unit(page, "tank1").locator(".ak-name")).to_have_text("First tank")
    expect(page.locator(f"{NODE}[data-kept='yes']")).to_have_count(10)
    expect(page.locator(EDGE)).to_have_count(11)
    expect(_unit(page, "tank1")).to_have_class(re.compile(r"\bselected\b"))


def test_a_unit_and_its_wires_are_one_removal(page):
    """Delete removes the unit and its two wires; one Undo brings all back."""
    _record_messages(page)
    _unit(page, "tank3").click()
    page.keyboard.press("Delete")
    _counts(page, 9, 9)
    expect(page.locator(EDGE)).to_have_count(9)
    assert _messages(page) == ["removed tank3: Undo brings it back"]
    _undo(page).click()
    _counts(page, 10, 11)
    expect(page.locator(EDGE)).to_have_count(11)
    expect(_undo(page)).to_be_disabled()


def test_backspace_does_not_delete(page):
    _unit(page, "tank3").click()
    page.keyboard.press("Backspace")
    page.wait_for_timeout(500)
    _counts(page, 10, 11)
    expect(page.locator(NODE)).to_have_count(10)


def test_a_removal_python_makes_is_not_echoed_back(page):
    """Undoing a dropped unit removes it from Python's list and so from the
    canvas; the canvas does not report that removal as an edit (which would
    leave something to undo)."""
    canvas = page.locator(".vue-flow__pane")
    page.get_by_text("CSTR", exact=True).first.drag_to(canvas, target_position={"x": 300, "y": 650})
    _counts(page, 11, 11)
    _undo(page).click()
    _counts(page, 10, 11)
    expect(page.locator(NODE)).to_have_count(10)
    page.wait_for_timeout(500)
    expect(_undo(page)).to_be_disabled()


def test_the_form_reaches_python_on_apply_only(page):
    """Typing changes nothing in Python; Apply sends the whole form."""
    summary = _unit(page, "tank1").locator(".ak-summary")
    expect(summary).to_have_text("volume 1000")
    _unit(page, "tank1").click()
    page.get_by_label("volume *").fill("1500")
    page.wait_for_timeout(1000)
    expect(summary).to_have_text("volume 1000")
    expect(_undo(page)).to_be_disabled()
    page.get_by_role("button", name="Apply").click()
    expect(summary).to_have_text("volume 1500")


def test_selecting_another_unit_applies_the_forms_edits_first(page):
    """Unapplied edits travel with the next selection and are applied to the
    unit they were typed for."""
    _unit(page, "tank1").click()
    page.get_by_label("volume *").fill("1700")
    _unit(page, "tank2").click()
    expect(_unit(page, "tank1").locator(".ak-summary")).to_have_text("volume 1700")
    expect(page.get_by_label("identifier")).to_have_value("tank2")


@pytest.mark.parametrize("bad", ["nan", "lots"])
def test_an_unreadable_edit_keeps_its_unit_and_names_the_field(page, bad):
    """Edits that cannot be read are not applied, and not dropped: the form
    stays on the unit they were typed for, with the field's error."""
    _unit(page, "tank1").click()
    page.get_by_label("volume *").fill(bad)
    _unit(page, "tank2").click()
    expect(page.get_by_text(re.compile(r"^tank1: volume: "))).to_be_visible()
    expect(page.get_by_label("identifier")).to_have_value("tank1")
    expect(page.get_by_label("volume *")).to_have_value(bad)
    expect(_unit(page, "tank1").locator(".ak-summary")).to_have_text("volume 1000")
