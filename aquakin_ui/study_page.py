"""The editor's Studies tab and its dialogs (study settings, results, the
question before replacing results), built with trame and Vuetify.

Each builder is bound onto :class:`~aquakin_ui.app.Editor` as a method (``self``
is the editor). The study dialog's draft is nested state, which the page does
not redraw on its own: see :mod:`aquakin_ui.repeated`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from trame.widgets import html
from trame.widgets import vuetify3 as v3

from aquakin_ui.repeated import add_button, bound, redraw, remove_button

if TYPE_CHECKING:
    from aquakin_ui.app import Editor


#: The state the worker gives a study whose ``study.yaml`` it cannot read.
UNREADABLE = "unreadable"
#: The colour of a study's state chip, for every state the worker reports (``None``
#: is the chip's own). The states are aquakin's (``check_study``, plus
#: ``UNREADABLE`` from ``list_studies``); ``tests/ui/test_protocol.py`` pins that
#: this lists exactly those, since the two sides share no import.
STATE_COLOURS = {
    "fresh": "success",
    "stale": "warning",
    "not run": None,
    "cancelled": None,
    "damaged": "error",
    "failed": "error",
    UNREADABLE: "error",
}


def _js(value) -> str:
    """``value`` (plain names: no quotes inside) as a JavaScript literal for a
    template expression. Single quotes, since the expression sits in a
    double-quoted HTML attribute, which a double quote would end."""
    text = json.dumps(value)
    if "'" in text:
        raise ValueError(f"{text} holds a quote and cannot go in a template")
    return text.replace('"', "'")


#: The state holding the study dialog's draft.
_STATE = "study_draft"
_REDRAW = redraw(_STATE)


def _bound(expr: str) -> dict:
    """Bind a widget to a value inside the study draft, redrawing after each change."""
    return bound(expr, _STATE)


def _studies_panel(self: Editor) -> None:
    """The Studies tab: its buttons, then a card per study of the project."""
    with html.Div(classes="d-flex align-center"):
        html.Div("Studies of this project", classes="text-overline")
        v3.VSpacer()
        v3.VBtn(
            icon="mdi-refresh",
            size="small",
            variant="text",
            click=self.refresh_studies,
            title="Check every study again",
        )
    v3.VBtn(
        "New study",
        prepend_icon="mdi-plus",
        block=True,
        variant="tonal",
        classes="text-none mb-3",
        click=self.new_study,
        disabled=("busy",),
    )
    v3.VSwitch(
        model_value=("ask_replace",),
        update_modelValue=(self.set_ask_replace, "[!!$event]"),
        label="Ask before replacing results",
        color="primary",
        density="compact",
        hide_details=True,
        classes="mb-2 ml-2",
        title="When off, running a study again replaces its results without asking",
    )
    html.Div(
        "No studies yet: set one up with New study (once the flowsheet has a file).",
        v_if="!studies.length",
        classes="text-body-2 text-medium-emphasis",
    )
    with v3.VCard(v_for="st in studies", key="st.name", variant="outlined", classes="mb-2"):
        _study_card_item()
        _study_card_actions(self)


#: A running study's latest progress (``st`` in a card): this editor's run's
#: own events, else what the run another program started records in its folder.
_PROGRESS = (
    "(study_progress[st.name] || (st.run && st.run.progress) || {fraction: null, message: ''})"
)


def _study_card_item() -> None:
    """A study card's heading, state, notes and run progress (the study is ``st``)."""
    with v3.VCardItem():
        with html.Div(classes="d-flex align-center ga-2"):
            html.Span("{{ st.name }}", classes="text-subtitle-2")
            v3.VChip("{{ st.kind }}", size="x-small", variant="tonal")
            v3.VIcon("mdi-snowflake", v_if="st.frozen", size="small", title="frozen")
            v3.VSpacer()
            v3.VChip(
                "{{ st.running ? 'running' : st.state }}",
                size="x-small",
                color=(f"st.running ? 'primary' : ({_js(STATE_COLOURS)}[st.state] ?? undefined)",),
            )
        html.Div(
            "{{ st.description }}",
            v_if="st.description",
            classes="text-caption text-medium-emphasis",
        )
        html.Div(
            "{{ r }}",
            v_for="r in st.reasons",
            key="r",
            classes="text-caption text-warning",
        )
        html.Div(
            "{{ n }}",
            v_for="n in st.notes",
            key="n",
            classes="text-caption text-medium-emphasis",
        )
        with html.Div(v_if="st.running", classes="mt-1"):
            v3.VProgressLinear(
                model_value=(f"{_PROGRESS}.fraction == null ? 0 : 100 * {_PROGRESS}.fraction",),
                indeterminate=(f"{_PROGRESS}.fraction == null",),
                height=4,
                rounded=True,
            )
            html.Div(
                f"{{{{ {_PROGRESS}.message }}}}",
                classes="text-caption text-medium-emphasis",
            )
            html.Div(
                "started by another program",
                v_if="!study_progress[st.name]",
                classes="text-caption text-medium-emphasis",
            )


def _study_card_actions(self: Editor) -> None:
    """A study card's buttons (the study is ``st``)."""
    with v3.VCardActions(classes="py-0 flex-wrap ga-0"):
        v3.VBtn(
            "Run",
            v_if="!st.running",
            size="small",
            disabled=(f"st.state === {_js(UNREADABLE)}",),
            click=(self.run_study, "[st.name, sel_dirty ? sel_rows : null]"),
        )
        v3.VBtn(
            "Cancel",
            v_else=True,
            size="small",
            color="error",
            click=(self.cancel_run, "[st.name]"),
        )
        v3.VBtn(
            "Results",
            v_if="!st.running && st.has_results",
            size="small",
            color="primary",
            click=(self.show_results, "[st.name]"),
        )
        v3.VBtn(
            "Freeze",
            v_if="!st.frozen",
            size="small",
            disabled=("st.running",),
            click=(self.freeze_study, "[st.name]"),
        )
        v3.VBtn(
            "Unfreeze",
            v_else=True,
            size="small",
            disabled=("st.running",),
            click=(self.unfreeze_study, "[st.name]"),
        )
        v3.VBtn(
            "Edit",
            size="small",
            disabled=(f"st.running || st.state === {_js(UNREADABLE)}",),
            click=(self.edit_study, "[st.name]"),
        )
        v3.VBtn("Copy", size="small", click=(self.copy_study, "[st.name]"))


def _observables_editor(rows: str) -> None:
    """Rows of ``{stream, channels}`` bound to the client expression ``rows``.

    A channel is a name the plant offers or a measure expression the user
    types; both travel as plain strings, and a structured measure read from
    a study file rides along in ``o.measures`` (see
    :mod:`aquakin_ui.study_form`).
    """
    with html.Div(v_for=f"(o, oi) in {rows}", key="oi", classes="d-flex ga-2 mb-1"):
        v3.VCombobox(
            **_bound("o.stream"),
            items=("study_choices.streams",),
            label="Stream",
            density="compact",
            variant="outlined",
            hide_details=True,
            style="max-width: 220px;",
        )
        # A combobox, not a select: a channel of the user's own is typed in
        # as a weighted sum ("BOD5 = 0.65*[SS] + 0.65*[XS]") and aquakin
        # reads it when the study is saved. The field's help says so.
        v3.VCombobox(
            **_bound("o.channels"),
            items=("study_choices.channels[o.stream] || []",),
            label="Channels (all species when empty; or type a sum of your own)",
            multiple=True,
            chips=True,
            closable_chips=True,
            density="compact",
            variant="outlined",
            hide_details=True,
        )
        remove_button(rows, "oi", _STATE)
    add_button("Add a stream", rows, "{stream: '', channels: [], measures: {}}", _STATE)


def _candidate_fields(c: str) -> None:
    with html.Div(classes="d-flex ga-2 mb-1"):
        v3.VTextField(
            **_bound(f"{c}.name"),
            label="Name",
            density="compact",
            variant="outlined",
            hide_details=True,
        )
        v3.VTextField(
            **_bound(f"{c}.sigma"),
            label="Measurement error (one, or one per channel)",
            density="compact",
            variant="outlined",
            hide_details=True,
        )
    _observables_editor(f"{c}.observables")


#: The keywords every input of the study dialog shares.
_COMMON = dict(density="compact", variant="outlined", hide_details=True)
#: The draft value of the field being drawn (``f``, in ``v_for="f in study_fields"``).
_DRAFT = "study_draft[f.name]"


def _study_settings_dialog(self: Editor) -> None:
    """The New / Edit study dialog, generated from aquakin's study form."""
    # Persistent: a click beside it would lose a long form; Cancel closes it.
    with v3.VDialog(
        v_model=("study_dialog_open",), max_width=760, scrollable=True, persistent=True
    ):
        with v3.VCard(title=("study_dialog_title",)):
            with v3.VCardText(style="max-height: 70vh;"):
                _study_heading(self)
                with html.Div(v_for="f in study_fields", key="f.name", classes="mb-3"):
                    _study_field()
            _study_dialog_actions(self)


def _study_heading(self: Editor) -> None:
    """The study's name and kind, and the kind's summary."""
    with html.Div(classes="d-flex ga-2 mb-1 pt-2"):
        v3.VTextField(
            v_model=("study_name",),
            label="Name",
            disabled=("!!study_editing",),
            **_COMMON,
        )
        v3.VSelect(
            model_value=("study_kind",),
            items=("study_kinds",),
            label="Kind",
            disabled=("!!study_editing",),
            update_modelValue=(self.set_study_kind, "[$event, study_draft]"),
            **_COMMON,
        )
    html.Div(
        "{{ (study_kinds.find(k => k.value === study_kind) || {}).summary }}",
        classes="text-caption text-medium-emphasis mb-3",
    )


def _study_field() -> None:
    """One field of the study form (``f``): its heading and the input for its type."""
    html.Div(
        "{{ f.label }}{{ f.required ? ' *' : '' }}",
        id=("'study-' + f.name + '-label'",),
        classes="text-subtitle-2",
    )
    html.Div("{{ f.help }}", classes="text-caption text-medium-emphasis mb-1")
    _plain_inputs()
    _span_input()
    _times_input()
    _run_time_input()
    _initial_state_input()
    _parameter_values_input()
    _transforms_input()
    with html.Div(v_if="f.type === 'observables'"):
        _observables_editor(_DRAFT)
    _candidates_input()
    _candidate_input()


def _plain_inputs() -> None:
    """The inputs that are one widget each: text, a box, a choice, parameters, streams."""
    v3.VTextField(
        v_if="['text', 'file', 'number', 'integer', 'numbers'].includes(f.type)",
        **_bound(_DRAFT),
        id=("'study-' + f.name",),
        **_COMMON,
    )
    v3.VCheckbox(
        v_if="f.type === 'boolean'",
        **_bound(_DRAFT),
        id=("'study-' + f.name",),
        label=("f.label",),
        **_COMMON,
    )
    v3.VSelect(
        v_if="f.type === 'choice'",
        **_bound(_DRAFT),
        id=("'study-' + f.name",),
        items=("f.choices",),
        **_COMMON,
    )
    v3.VCombobox(
        v_if="f.type === 'parameter'",
        **_bound(_DRAFT),
        id=("'study-' + f.name",),
        items=("study_choices.parameters",),
        **_COMMON,
    )
    v3.VCombobox(
        v_if="['streams', 'parameters'].includes(f.type)",
        **_bound(_DRAFT),
        id=("'study-' + f.name",),
        items=("f.type === 'streams' ? study_choices.streams : study_choices.parameters",),
        multiple=True,
        chips=True,
        closable_chips=True,
        **_COMMON,
    )


def _span_input() -> None:
    """A from / to pair."""
    with html.Div(v_if="f.type === 'span'", classes="d-flex ga-2"):
        v3.VTextField(**_bound(f"{_DRAFT}.start"), label="From", **_COMMON)
        v3.VTextField(**_bound(f"{_DRAFT}.stop"), label="To", **_COMMON)


def _times_input() -> None:
    """Times as an even grid (from, to, how many) or a list."""
    with html.Div(v_if="f.type === 'times'"):
        with v3.VBtnToggle(**_bound(f"{_DRAFT}.mode"), density="compact", mandatory=True):
            v3.VBtn("Evenly spaced", value="grid", classes="text-none")
            v3.VBtn("A list", value="list", classes="text-none")
        with html.Div(v_if=f"{_DRAFT}.mode === 'grid'", classes="d-flex ga-2 mt-1"):
            v3.VTextField(**_bound(f"{_DRAFT}.start"), label="From", **_COMMON)
            v3.VTextField(**_bound(f"{_DRAFT}.stop"), label="To", **_COMMON)
            v3.VTextField(**_bound(f"{_DRAFT}.num"), label="How many", **_COMMON)
        v3.VTextField(
            v_else=True,
            **_bound(f"{_DRAFT}.list"),
            label="Values, separated by commas",
            classes="mt-1",
            **_COMMON,
        )


def _run_time_input() -> None:
    """The steady state, or a dynamic run's span, time unit and saved times."""
    with html.Div(v_if="f.type === 'run_time'"):
        v3.VCheckbox(
            **_bound(f"{_DRAFT}.dynamic"),
            label="A dynamic run (otherwise the steady state)",
            **_COMMON,
        )
        with html.Div(v_if=f"{_DRAFT}.dynamic"):
            with html.Div(classes="d-flex ga-2 mb-1"):
                v3.VTextField(**_bound(f"{_DRAFT}.start"), label="From", **_COMMON)
                v3.VTextField(**_bound(f"{_DRAFT}.stop"), label="To", **_COMMON)
                v3.VTextField(**_bound(f"{_DRAFT}.unit"), label="Time unit", **_COMMON)
            with html.Div(classes="d-flex ga-2"):
                v3.VTextField(
                    **_bound(f"{_DRAFT}.every.start"),
                    label="Save from",
                    **_COMMON,
                )
                v3.VTextField(**_bound(f"{_DRAFT}.every.stop"), label="to", **_COMMON)
                v3.VTextField(**_bound(f"{_DRAFT}.every.num"), label="times", **_COMMON)


def _initial_state_input() -> None:
    """Where a dynamic run starts: each unit's default, the steady state or a file."""
    with html.Div(v_if="f.type === 'initial_state'", classes="d-flex ga-2"):
        v3.VSelect(
            **_bound(f"{_DRAFT}.mode"),
            items=(
                "[{title: 'the default of each unit', value: 'default'}, "
                "{title: 'the steady state', value: 'steady'}, "
                "{title: 'a saved state', value: 'file'}]",
            ),
            **_COMMON,
        )
        v3.VTextField(
            v_if=f"{_DRAFT}.mode === 'file'",
            **_bound(f"{_DRAFT}.file"),
            label="File (from the project folder)",
            **_COMMON,
        )


def _parameter_values_input() -> None:
    """Rows of a parameter and its value."""
    with html.Div(v_if="f.type === 'parameter_values'"):
        with html.Div(v_for=f"(r, ri) in {_DRAFT}", key="ri", classes="d-flex ga-2 mb-1"):
            v3.VCombobox(
                **_bound("r.name"),
                items=("study_choices.parameters",),
                label="Parameter",
                **_COMMON,
            )
            v3.VTextField(
                **_bound("r.value"),
                label="Value",
                style="max-width: 160px;",
                **_COMMON,
            )
            remove_button(_DRAFT, "ri", _STATE)
        add_button("Add a parameter", _DRAFT, "{name: '', value: ''}", _STATE)


def _transforms_input() -> None:
    """Rows of a parameter and the transform it is estimated in."""
    with html.Div(v_if="f.type === 'transforms'"):
        with html.Div(v_for=f"(r, ri) in {_DRAFT}", key="ri", classes="d-flex ga-2 mb-1"):
            v3.VCombobox(
                **_bound("r.name"),
                items=("study_choices.parameters",),
                label="Parameter",
                **_COMMON,
            )
            v3.VSelect(
                **_bound("r.transform"),
                items=("f.choices",),
                style="max-width: 180px;",
                **_COMMON,
            )
            remove_button(_DRAFT, "ri", _STATE)
        add_button("Add a transform", _DRAFT, "{name: '', transform: 'positive_log'}", _STATE)


def _candidates_input() -> None:
    """Sensor sets to rank, each a name, a measurement error and its streams."""
    with html.Div(v_if="f.type === 'candidates'"):
        with v3.VSheet(
            v_for=f"(c, ci) in {_DRAFT}",
            key="ci",
            border=True,
            rounded=True,
            classes="pa-2 mb-2",
        ):
            _candidate_fields("c")
            v3.VBtn(
                "Remove this set",
                size="small",
                variant="text",
                color="error",
                classes="text-none",
                click=f"{_DRAFT}.splice(ci, 1){_REDRAW}",
            )
        v3.VBtn(
            "Add a sensor set",
            prepend_icon="mdi-plus",
            size="small",
            variant="tonal",
            classes="text-none",
            click=(
                f"{_DRAFT}.push({{name: '', sigma: '', "
                "observables: [{stream: '', channels: [], measures: {}}]})"
                f"{_REDRAW}"
            ),
        )


def _candidate_input() -> None:
    """The sensors already in place, when there are any."""
    with html.Div(v_if="f.type === 'candidate'"):
        v3.VCheckbox(
            **_bound(f"{_DRAFT}.enabled"),
            label="There are sensors in place",
            **_COMMON,
        )
        with html.Div(v_if=f"{_DRAFT}.enabled"):
            _candidate_fields(_DRAFT)


def _study_dialog_actions(self: Editor) -> None:
    """The error, beside the buttons rather than at the foot of the scrolled form,
    then Cancel and Create / Save."""
    v3.VAlert(
        "{{ study_error }}",
        v_if="study_error",
        type="error",
        variant="tonal",
        density="compact",
        classes="text-caption mx-4 flex-grow-0",
        style="white-space: pre-wrap;",
    )
    with v3.VCardActions():
        v3.VSpacer()
        v3.VBtn(
            "Cancel",
            variant="text",
            classes="text-none",
            click="study_dialog_open = false",
        )
        v3.VBtn(
            "{{ study_editing ? 'Save' : 'Create' }}",
            color="primary",
            variant="flat",
            classes="text-none",
            loading=("busy",),
            click=(
                self.save_study_dialog,
                "[study_name, study_kind, study_draft]",
            ),
        )


def _table(table: str) -> None:
    """A results table bound to the state ``table`` (``steady_table``'s shape)."""
    with v3.VTable(density="compact", classes="mb-2"):
        with html.Thead():
            with html.Tr():
                html.Th("")
                html.Th("Unit")
                html.Th("{{ c }}", v_for=f"c in {table}.columns", key="c", classes="ak-num")
        with html.Tbody():
            with html.Tr(v_for=f"r in {table}.rows", key="r.channel"):
                html.Td("{{ r.channel }}", classes="font-weight-medium")
                html.Td("{{ r.unit }}", classes="text-medium-emphasis text-caption")
                html.Td("{{ v }}", v_for="(v, vi) in r.values", key="vi", classes="ak-num")


def _results_dialog(self: Editor) -> None:
    with v3.VDialog(v_model=("results_open",), max_width=980, scrollable=True):
        with v3.VCard():
            v3.VCardTitle("{{ results_title }}", classes="pt-4")
            with v3.VCardText():
                v3.VAlert(
                    "{{ results_note }}",
                    v_if="results_note",
                    type="warning",
                    variant="tonal",
                    density="compact",
                    classes="mb-3 text-body-2",
                )
                with html.Div(v_if="results_mode === 'steady'"):
                    with html.Div(v_if="results_measured.rows.length"):
                        html.Div("Measured", classes="text-subtitle-2")
                        _table("results_measured")
                        with v3.VExpansionPanels(variant="accordion", classes="mb-2"):
                            with v3.VExpansionPanel(title="Flow and species"):
                                with v3.VExpansionPanelText():
                                    _table("results_streams")
                    with html.Div(v_else=True):
                        html.Div("Flow and species", classes="text-subtitle-2")
                        _table("results_streams")
                with html.Div(v_if="results_mode === 'time'"):
                    with html.Div(classes="d-flex ga-2 mb-3"):
                        v3.VSelect(
                            label="Stream",
                            model_value=("results_stream",),
                            items=("results_stream_choices",),
                            update_modelValue=(self.choose_results_stream, "[$event]"),
                            density="compact",
                            variant="outlined",
                            hide_details=True,
                            style="max-width: 240px;",
                        )
                        v3.VSelect(
                            label="Channels",
                            model_value=("results_channels",),
                            items=("results_channel_choices",),
                            update_modelValue=(self.show_charts, "[$event]"),
                            multiple=True,
                            chips=True,
                            closable_chips=True,
                            density="compact",
                            variant="outlined",
                            hide_details=True,
                        )
                    with html.Div(classes="ak-charts"):
                        html.Div(v_for="c in results_charts", key="c.channel", v_html="c.svg")
                with html.Div(v_if="results_mode === 'analysis'"):
                    html.P(
                        "{{ line }}",
                        v_for="line in results_lines",
                        key="line",
                        classes="text-body-2 mb-2",
                    )
                    with html.Div(v_for="tb in results_tables", key="tb.title", classes="mb-3"):
                        html.Div("{{ tb.title }}", classes="text-subtitle-2")
                        with v3.VTable(density="compact"):
                            with html.Thead():
                                with html.Tr():
                                    html.Th(
                                        "{{ c }}",
                                        v_for="(c, ci) in tb.columns",
                                        key="ci",
                                        classes=("ci ? 'ak-num' : ''",),
                                    )
                            with html.Tbody():
                                with html.Tr(v_for="(row, ri) in tb.rows", key="ri"):
                                    html.Td(
                                        "{{ cell }}",
                                        v_for="(cell, ci) in row",
                                        key="ci",
                                        classes=("ci ? 'ak-num' : 'font-weight-medium'",),
                                    )
                    with html.Div(classes="ak-charts"):
                        html.Div(v_for="c in results_charts", key="c.channel", v_html="c.svg")
                    html.Div(
                        "Files: {{ results_files.join(', ') }}",
                        classes="text-caption text-medium-emphasis mt-2",
                    )
                html.Div(
                    "Saved in {{ results_folder }}",
                    classes="text-caption text-medium-emphasis mt-2",
                )
            with v3.VCardActions():
                v3.VSpacer()
                v3.VBtn("Close", variant="text", classes="text-none", click="results_open = false")


def _study_dialog(self: Editor) -> None:
    with v3.VDialog(v_model=("study_confirm_open",), max_width=560, persistent=True):
        with v3.VCard(title=("study_confirm_title",)):
            with v3.VCardText():
                html.Div("{{ study_confirm_text }}")
                v3.VCheckbox(
                    v_if="study_confirm_kind === 'run'",
                    v_model=("study_confirm_dont_ask",),
                    label="Don't ask again before replacing results",
                    hint="You can turn this back on in the Studies tab.",
                    persistent_hint=True,
                    density="compact",
                    classes="mt-2",
                )
            with v3.VCardActions(classes="flex-wrap ga-1"):
                v3.VSpacer()
                v3.VBtn(
                    "Cancel",
                    click=(self.answer_study, "['cancel']"),
                    variant="text",
                    classes="text-none",
                )
                v3.VBtn(
                    "Copy first",
                    click=(self.answer_study, "['copy']"),
                    variant="tonal",
                    classes="text-none",
                )
                v3.VBtn(
                    "{{ study_confirm_label }}",
                    click=(self.answer_study, "['go']"),
                    color="error",
                    variant="flat",
                    classes="text-none",
                )
