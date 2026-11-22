"""Where the canvas validates a wire being drawn, read from the widget's source.

The check that refuses a wire into a fed input (or out of a used output) must
sit on the handles (``UnitNode.vue``), never on Vue Flow's canvas-wide
validator: Vue Flow runs that one on every edge added to a populated canvas,
and the check refuses an edge whose handle another edge uses, so it would
drop wires Python holds (``.claude/rules/ui.md``; the behaviour is pinned in
``tests/browser``). This reads the source, so it needs no browser; the
``widget`` CI job ties the committed bundle to the same source.
"""

import re
from pathlib import Path

SRC = Path(__file__).parents[2] / "ui-widget" / "src"

#: Every spelling Vue accepts for the prop, reduced by ``_key``.
VALIDATOR = "isvalidconnection"
#: One attribute of an opening tag (name and optional value), or the tag's end.
_ATTRIBUTE = re.compile(r"""\s*(/?>|([^\s=/>]+)(\s*=\s*("[^"]*"|'[^']*'|[^\s>]+))?)""")


def _key(name):
    """A tag or attribute name without binding prefix, dashes or case:
    ``:is-valid-connection``, ``v-bind:isValidConnection`` -> ``isvalidconnection``."""
    name = re.sub(r"^(v-bind:|:)", "", name)
    return name.replace("-", "").replace("_", "").lower()


def _tags(source, name):
    """The opening tags named ``name`` (any spelling) in a component's source,
    each as ``{attribute key: raw attribute name}``. Attribute values are
    skipped with their quotes, so a ``>`` inside one does not end the tag."""
    found = []
    for m in re.finditer(r"<([A-Za-z][\w-]*)", source):
        if _key(m.group(1)) != _key(name):
            continue
        attrs, i = {}, m.end()
        while True:
            part = _ATTRIBUTE.match(source, i)
            if part is None:
                raise AssertionError(f"cannot read the <{m.group(1)}> tag at {source[i:i + 40]!r}")
            i = part.end()
            if part.group(2) is None:
                break
            attrs[_key(part.group(2))] = part.group(2)
        found.append(attrs)
    return found


def _source(name):
    return (SRC / name).read_text(encoding="utf-8")


def test_reading_a_tag_finds_the_prop_in_any_spelling_and_order():
    """The reader this file relies on: attribute order, kebab or camel case,
    ``:`` or ``v-bind:``, and a ``>`` inside a value do not hide the prop."""
    tags = _tags(
        '<vue-flow :mini="(n) => n > 1" v-bind:isValidConnection="ok" id=x />'
        '<VueFlow\n  :is-valid-connection="ok"\n  :x="1">'
        "<VueFlow :x='a > b'></VueFlow>",
        "VueFlow",
    )
    assert [VALIDATOR in t for t in tags] == [True, True, False]
    assert tags[0]["id"] == "id"


def test_the_canvas_passes_vue_flow_no_connection_validator():
    """No component that renders ``<VueFlow>`` names a connection validator,
    in its template or its script (``useVueFlow({isValidConnection})`` or the
    store's ``isValidConnection`` would be the same canvas-wide check)."""
    canvases = [p for p in sorted(SRC.glob("*.vue")) if _tags(_source(p.name), "VueFlow")]
    assert [p.name for p in canvases] == ["FlowsheetCanvas.vue"]
    for path in canvases:
        mentions = [
            w for w in re.findall(r"[\w-]+", _source(path.name)) if _key(w) == VALIDATOR
        ]
        assert mentions == [], f"{path.name} gives Vue Flow a connection validator"


def test_every_handle_carries_the_wire_check():
    """Each ``<Handle>`` in the widget binds the check (a plain string
    attribute would not be a function), so a wire is judged while it is drawn;
    the unit draws two, one for its inputs and one for its outputs."""
    handles = {p.name: _tags(_source(p.name), "Handle") for p in sorted(SRC.glob("*.vue"))}
    assert {name: len(h) for name, h in handles.items() if h} == {"UnitNode.vue": 2}
    for h in handles["UnitNode.vue"]:
        assert h.get(VALIDATOR, "").startswith((":", "v-bind:")), h
