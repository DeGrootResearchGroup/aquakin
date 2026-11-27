"""The API reference as ``aquakin mcp`` serves it: public names, signatures and
docstrings read from the installed package."""

from __future__ import annotations

import inspect

import pytest

import aquakin
import aquakin.plant
import aquakin.study
from aquakin.mcp.api import MODULES, _signature, describe_api, list_api


@pytest.mark.parametrize("module", [aquakin, aquakin.plant, aquakin.study])
def test_list_api_is_the_modules_all_in_order(module):
    listed = list_api(module.__name__)
    assert [entry["name"] for entry in listed] == list(module.__all__)
    assert module.__name__ in MODULES


def test_list_api_gives_each_names_kind_and_first_docstring_paragraph():
    entries = {e["name"]: e for e in list_api()}
    assert entries["Plant"]["kind"] == "class"
    assert entries["calibrate"]["kind"] == "function"
    first = inspect.getdoc(aquakin.calibrate).split("\n\n")[0]
    assert entries["calibrate"]["summary"] == " ".join(first.split())
    data = {e["name"]: e for e in list_api("aquakin.plant")}["A2O_WARM_REACTOR_COMPOSITION"]
    assert data == {"name": "A2O_WARM_REACTOR_COMPOSITION", "kind": "data", "summary": ""}


def test_an_unknown_module_lists_the_api_modules():
    with pytest.raises(KeyError, match=r"no API module 'aquakin.core'; the modules are \["):
        list_api("aquakin.core")


@pytest.mark.parametrize(
    "name, full, obj",
    [
        ("calibrate", "aquakin.calibrate", aquakin.calibrate),
        ("Plant.calibrate", "aquakin.Plant.calibrate", aquakin.Plant.calibrate),
        ("aquakin.study.run_study", "aquakin.study.run_study", aquakin.study.run_study),
        (
            "aquakin.plant.bsm.build_bsm1",
            "aquakin.plant.bsm.build_bsm1",
            aquakin.plant.bsm.build_bsm1,
        ),
        ("IFASUnit", "aquakin.plant.IFASUnit", aquakin.plant.IFASUnit),
    ],
)
def test_describe_api_gives_the_signature_and_full_docstring(name, full, obj):
    described = describe_api(name)
    assert described["name"] == full
    assert described["signature"] == str(inspect.signature(obj))
    assert described["doc"] == inspect.getdoc(obj)


def test_a_class_lists_its_public_aquakin_members():
    members = {m["name"]: m for m in describe_api("Plant")["members"]}
    assert members["calibrate"]["kind"] == "function"
    assert members["time_unit"]["kind"] == "property"
    assert not [m for m in members if m.startswith("_")]


def test_members_inherited_from_outside_aquakin_are_left_out():
    # StudyError's methods (with_traceback, add_note, ...) all come from
    # BaseException.
    assert describe_api("aquakin.study.StudyError")["members"] == []


def test_a_property_is_described_as_itself():
    described = describe_api("Plant.time_unit")
    prop = inspect.getattr_static(aquakin.Plant, "time_unit")
    assert described["kind"] == "property"
    assert described["doc"] == inspect.getdoc(prop)
    assert described["signature"] == str(inspect.signature(prop.fget))


def test_a_module_and_a_constant_are_described():
    assert describe_api("aquakin")["kind"] == "module"
    assert describe_api("aquakin")["signature"] is None
    constant = describe_api("aquakin.plant.A2O_WARM_REACTOR_COMPOSITION")
    assert constant["kind"] == "data"
    assert constant["doc"] == repr(aquakin.plant.A2O_WARM_REACTOR_COMPOSITION)


def test_a_callable_without_an_introspectable_signature_has_none():
    # ``max`` is a builtin whose signature inspect cannot read (ValueError).
    with pytest.raises(ValueError, match="no signature found"):
        inspect.signature(max)
    assert _signature(max) is None
    assert _signature(len) == "(obj, /)"


@pytest.mark.parametrize(
    "name, message",
    [
        ("_private", r"'_private' is not a public name"),
        ("aquakin.plant._builder_support", r"is not a public name"),
        ("Plant..calibrate", r"is not a public name"),
        ("not_a_name", r"'not_a_name' is not in the public API of \["),
        ("Plant.not_a_member", r"aquakin.Plant has no public attribute 'not_a_member'"),
        ("aquakin.plant.not_here", r"aquakin.plant has no public attribute 'not_here'"),
    ],
)
def test_a_private_or_missing_name_is_refused(name, message):
    with pytest.raises(KeyError, match=message):
        describe_api(name)
