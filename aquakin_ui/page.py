"""The editor's page: the layout, toolbar, palette, side panel, unit form and
the dialogs about the flowsheet, built with trame and Vuetify.

Each builder is bound onto :class:`~aquakin_ui.app.Editor` as a method (``self``
is the editor); the Studies tab and the study dialogs are in
:mod:`aquakin_ui.study_page`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import client, html
from trame.widgets import vuetify3 as v3

from aquakin_ui.repeated import add_button, bound, move_button, remove_button
from aquakin_ui.widget import Flowsheet as FlowsheetCanvas

if TYPE_CHECKING:
    from aquakin_ui.app import Editor


DRAG_TYPE = "application/x-aquakin-unit"

PAGE_CSS = """
.ak-chart { display: block; font: 11px Roboto, sans-serif; }
.ak-chart-title { font-size: 13px; font-weight: 500; fill: rgba(var(--v-theme-on-surface), .87); }
.ak-chart-axis { fill: rgba(var(--v-theme-on-surface), .62); }
.ak-chart-grid { stroke: rgba(var(--v-theme-on-surface), .12); stroke-width: 1; }
.ak-chart-line { fill: none; stroke: #2a78d6; stroke-width: 2; stroke-linejoin: round;
  stroke-linecap: round; }
.ak-chart-hit rect { fill: transparent; }
.ak-chart-hit line, .ak-chart-hit circle { opacity: 0; }
.ak-chart-hit line { stroke: rgba(var(--v-theme-on-surface), .35); stroke-width: 1; }
.ak-chart-hit circle { fill: #2a78d6; stroke: rgb(var(--v-theme-surface)); stroke-width: 2; }
.ak-chart-hit:hover line, .ak-chart-hit:hover circle { opacity: 1; }
.v-theme--dark .ak-chart-line { stroke: #3987e5; }
.v-theme--dark .ak-chart-hit circle { fill: #3987e5; }
.ak-chart-dot { fill: #eb6834; stroke: rgb(var(--v-theme-surface)); stroke-width: 2; }
.v-theme--dark .ak-chart-dot { fill: #d95926; }
.ak-chart-ref { stroke: rgba(var(--v-theme-on-surface), .55); stroke-width: 1; }
.ak-num { font-variant-numeric: tabular-nums; text-align: right !important; white-space: nowrap; }
.ak-charts { display: grid; gap: 16px; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); }
.ak-pal-item { cursor: grab; user-select: none; display: flex; align-items: center; gap: 10px;
  padding: 5px 10px; margin: 1px 8px; border-radius: 6px; font-size: 13px; }
.ak-pal-item:hover { background: rgba(127,127,127,.12); }
.ak-pal-icon { width: 20px; height: 20px; flex: none; display: inline-flex; }
.ak-pal-icon svg { width: 20px; height: 20px; }
.ak-sel-icon svg { width: 28px; height: 28px; }
"""


def _build(self: Editor) -> None:
    # The form's rows are edited in the browser and sent whole on Apply, so the
    # page's own changes to them (a composition's entries redrawn) are not sent
    # back: Python's last rows stay what a redraw is compared with, so an
    # unchanged background redraw still leaves the typing alone.
    self.state.client_only("sel_rows")
    with SinglePageWithDrawerLayout(self.server, theme=("dark ? 'dark' : 'light'",)) as layout:
        layout.title.set_text("aquakin · {{ plant_name }}")
        layout.footer.hide()
        self._toolbar(layout)
        self._palette(layout)
        with layout.content:
            with html.Div(style="display: flex; height: calc(100vh - 64px);"):
                with html.Div(style="flex: 1; position: relative;"):
                    FlowsheetCanvas(
                        nodes=("nodes",),
                        edges=("edges",),
                        icons=("icons",),
                        drag_type=DRAG_TYPE,
                        fit_key=("fit_key",),
                        connect=(self.on_connect, "[$event]"),
                        move=(self.on_move, "[$event]"),
                        remove=(self.on_remove, "[$event]"),
                        dropped=(self.on_dropped, "[$event]"),
                        select=(self.on_select, "[$event, sel_dirty ? sel_rows : null]"),
                        style="position: absolute; inset: 0;",
                    )
                self._side_panel()
            self._confirm_dialog()
            self._study_dialog()
            self._study_settings_dialog()
            self._results_dialog()
            self._unsaved_dialog()
            v3.VSnackbar(
                "{{ message }}",
                v_model=("show_message",),
                timeout=4000,
                color=("message_colour",),
            )


def _toolbar(self: Editor, layout) -> None:
    with layout.toolbar:
        v3.VSpacer()
        v3.VTextField(
            v_model=("spec_path",),
            placeholder="path/to/flowsheet.yaml",
            title="The flowsheet's file: Open reads it, Save writes it",
            density="compact",
            variant="outlined",
            hide_details=True,
            prepend_inner_icon="mdi-file-outline",
            style="max-width: 300px;",
            classes="mr-2",
        )
        v3.VBtn(
            icon="mdi-undo",
            click=self.undo,
            disabled=("!can_undo",),
            variant="text",
            title="Undo the last edit of the flowsheet",
            classes="mr-1",
        )
        v3.VBtn(
            "Open",
            click=(self.request_open, "[sel_dirty ? sel_rows : null]"),
            disabled=("busy",),
            variant="tonal",
            classes="mr-1 text-none",
        )
        v3.VBtn(
            "Save",
            click=(self.save_button, "[sel_dirty ? sel_rows : null]"),
            loading=("busy",),
            disabled=("busy",),
            variant="tonal",
            classes="mr-2 text-none",
        )
        v3.VBtn(
            "Simulate",
            prepend_icon="mdi-play",
            click=(self.simulate, "[sel_dirty ? sel_rows : null]"),
            loading=("simulating",),
            disabled=("busy",),
            color="primary",
            variant="flat",
            classes="mr-4 text-none",
            title="Run the flowsheet's simulation study and show what leaves the plant",
        )
        v3.VChip(
            "{{ nodes.length }} units · {{ edges.length }} connections · {{ n_recycles }} recycles",
            size="small",
            variant="tonal",
            classes="mr-2",
        )
        v3.VBtn(icon="mdi-theme-light-dark", click="dark = !dark", variant="text")


def _palette(self: Editor, layout) -> None:
    with layout.drawer as drawer:
        drawer.width = 250
        client.Style(PAGE_CSS)
        html.Div(
            "Drag a unit onto the canvas",
            classes="text-caption text-medium-emphasis px-4 pt-3 pb-1",
        )
        with html.Div(v_for="group in palette", key="group.category", classes="pb-1"):
            html.Div(
                "{{ group.category }}", classes="text-overline px-4", style="line-height: 2rem;"
            )
            with html.Div(
                v_for="k in group.kinds",
                key="k.kind",
                classes="ak-pal-item",
                draggable="true",
                title=("k.summary",),
                __events=["dragstart"],
                dragstart=f"$event.dataTransfer.setData('{DRAG_TYPE}', k.kind); "
                "$event.dataTransfer.effectAllowed = 'copy'",
            ):
                html.Span(v_html="k.icon", classes="ak-pal-icon", style=("{ color: k.colour }",))
                html.Span("{{ k.label }}")
                html.Span(
                    "Python",
                    v_if="k.python",
                    classes="text-caption text-medium-emphasis ms-auto",
                    title="Some of its settings can only be set from Python",
                )


def _side_panel(self: Editor) -> None:
    with v3.VSheet(width=340, border="s", style="overflow: auto; flex: none;"):
        with v3.VTabs(v_model=("panel",), density="compact", grow=True):
            v3.VTab("Unit", value="properties", classes="text-none", min_width=0)
            v3.VTab(
                "Checks ({{ errors.length }})", value="checks", classes="text-none", min_width=0
            )
            v3.VTab("Studies", value="studies", classes="text-none", min_width=0)
            v3.VTab("Spec", value="spec", classes="text-none", min_width=0)
        with v3.VWindow(v_model=("panel",)):
            with v3.VWindowItem(value="studies", classes="pa-3"):
                self._studies_panel()
            with v3.VWindowItem(value="properties", classes="pa-3"):
                self._unit_form()
            with v3.VWindowItem(value="checks", classes="pa-3"):
                html.Div("Errors", classes="text-overline")
                with html.Div(v_if="checking", classes="d-flex align-center ga-2 mb-2"):
                    v3.VProgressCircular(indeterminate=True, size=16, width=2)
                    html.Span("aquakin is checking the flowsheet…", classes="text-body-2")
                html.Div(
                    "None: aquakin built every unit and the plant.",
                    v_if="!checking && checked && !errors.length",
                    classes="text-body-2 mb-2",
                )
                with v3.VList(density="compact", v_if="errors.length"):
                    v3.VListItem(
                        v_for="c in errors",
                        key="c",
                        subtitle=("c",),
                        prepend_icon="mdi-alert-circle-outline",
                        base_color="error",
                        lines="three",
                    )
                html.Div("Product streams (unconnected outputs)", classes="text-overline mt-2")
                with v3.VList(density="compact"):
                    v3.VListItem(
                        v_for="p in products",
                        key="p",
                        title=("p",),
                        prepend_icon="mdi-arrow-right-bold-outline",
                    )
            with v3.VWindowItem(value="spec", classes="pa-3"):
                html.Div("The spec as it will be saved", classes="text-caption mb-1")
                html.Pre(
                    "{{ spec_text }}",
                    style="font-size: 11px; white-space: pre; overflow-x: auto;",
                )


def _unsaved_dialog(self: Editor) -> None:
    with v3.VDialog(v_model=("unsaved_open",), max_width=520, persistent=True):
        with v3.VCard(title="Unsaved edits"):
            v3.VCardText("{{ unsaved_text }}")
            with v3.VCardActions(classes="flex-wrap ga-1"):
                v3.VSpacer()
                v3.VBtn(
                    "Cancel",
                    click=(self.answer_unsaved, "['cancel']"),
                    variant="text",
                    classes="text-none",
                )
                v3.VBtn(
                    "Discard edits",
                    click=(self.answer_unsaved, "['discard']"),
                    color="error",
                    variant="tonal",
                    classes="text-none",
                )
                v3.VBtn(
                    "Save first",
                    v_if="unsaved_can_save",
                    click=(self.answer_unsaved, "['save']"),
                    color="primary",
                    variant="flat",
                    classes="text-none",
                )


def _confirm_dialog(self: Editor) -> None:
    with v3.VDialog(v_model=("confirm_open",), max_width=480, persistent=True):
        with v3.VCard(title="Remove connections?"):
            with v3.VCardText():
                html.Div(
                    "These settings remove ports that are connected. Applying them "
                    "removes these connections:",
                    classes="mb-2",
                )
                html.Div("{{ c }}", v_for="c in confirm_lines", key="c", classes="text-mono")
            with v3.VCardActions():
                v3.VSpacer()
                v3.VBtn("Cancel", click=self.cancel_apply, variant="text")
                v3.VBtn(
                    "Apply and remove",
                    click=self.confirm_apply,
                    color="error",
                    variant="flat",
                )


def _unit_form(self: Editor) -> None:
    html.Div(
        "Select a unit to edit it.",
        v_if="!selected",
        classes="text-body-2 text-medium-emphasis",
    )
    with html.Div(v_else=True):
        with html.Div(classes="d-flex align-center ga-2 mb-1"):
            html.Span(v_html="sel_icon", classes="ak-sel-icon d-inline-flex")
            html.Div("{{ sel_kind }}", classes="text-subtitle-1 font-weight-medium")
        html.Div("{{ sel_summary }}", classes="text-caption text-medium-emphasis mb-3")
        v3.VBtn(
            "{{ sel_dirty ? 'Apply changes' : 'Apply' }}",
            click=(self.apply_properties, "[sel_rows]"),
            color="primary",
            block=True,
            variant=("sel_dirty ? 'flat' : 'tonal'",),
            classes="mb-3 text-none",
        )
        html.Div(
            "Selecting another unit applies these changes.",
            v_if="sel_dirty",
            classes="text-caption text-medium-emphasis mb-2",
        )
        v3.VAlert(
            "{{ sel_error }}",
            v_if="sel_error",
            type="error",
            variant="tonal",
            density="compact",
            classes="mb-3 text-caption",
        )
        with html.Div(v_for="(row, i) in sel_rows", key="row.key"):
            common = dict(
                label=("row.label",),
                hide_details="auto",
                update_focused="if ($event) { focus_label = row.label; focus_help = row.hint }",
                placeholder=("row.placeholder",),
                persistent_placeholder=True,
                density="compact",
                variant="outlined",
                classes="mb-2",
            )
            v3.VSelect(
                v_if="row.widget === 'select' && row.reshape",
                items=("row.items",),
                model_value=("row.value",),
                update_modelValue=(self.reshape_form, "[sel_rows, i, $event]"),
                **common,
            )
            v3.VSelect(
                v_else_if="row.widget === 'select'",
                items=("row.items",),
                v_model="row.value",
                update_modelValue="sel_dirty = true",
                **common,
            )
            v3.VSelect(
                v_else_if="row.widget === 'multi'",
                items=("row.items",),
                multiple=True,
                chips=True,
                closable_chips=True,
                v_model="row.value",
                update_modelValue="sel_dirty = true",
                **common,
            )
            _composition_input(common)
            _records_input()
            v3.VTextField(
                v_else_if="row.widget === 'readonly'",
                disabled=True,
                model_value=("row.value",),
                **common,
            )
            v3.VTextField(
                v_else=True,
                v_model="row.value",
                update_modelValue="sel_dirty = true",
                suffix=("row.suffix",),
                error_messages=("sel_errors[row.key] || []",),
                **common,
            )
        with v3.VSheet(
            v_if="focus_help", rounded=True, border=True, classes="pa-2 mt-1 text-caption"
        ):
            html.Div("{{ focus_label }}", classes="font-weight-medium mb-1")
            html.Div("{{ focus_help }}", classes="text-medium-emphasis")


#: A form row changed in the browser: Apply turns "Apply changes".
_DIRTY = "sel_dirty = true"


def _composition_input(common: dict) -> None:
    """A composition (``row.value``, entries of a species and its concentration),
    each species chosen from ``row.items``."""
    with html.Div(v_else_if="row.widget === 'composition'", classes="ak-composition mb-2"):
        html.Div("{{ row.label }}", classes="text-caption text-medium-emphasis mb-1")
        compact = {k: v for k, v in common.items() if k not in ("label", "placeholder", "classes")}
        with html.Div(v_for="(e, ei) in row.value", key="ei", classes="d-flex ga-1 mb-1"):
            v3.VSelect(
                **bound("e.species", "sel_rows", _DIRTY),
                items=("row.items",),
                label="species",
                style="flex: 0 0 104px;",
                **compact,
            )
            v3.VTextField(
                **bound("e.value", "sel_rows", _DIRTY),
                label="concentration",
                suffix=("row.units[e.species] || ''",),
                **compact,
            )
            remove_button("row.value", "ei", "sel_rows", _DIRTY)
        html.Div(
            "{{ sel_errors[row.key] }}",
            v_if="sel_errors[row.key]",
            classes="text-caption text-error mb-1",
        )
        add_button("Add a species", "row.value", "{species: '', value: ''}", "sel_rows", _DIRTY)


def _records_input() -> None:
    """A table of records (``row.value``, an SBR's phases): one card per record
    with a widget per field of ``row.fields``, moved up or down or removed by
    its buttons; a box for each true / false field."""
    field = dict(
        density="compact",
        variant="outlined",
        hide_details=True,
        update_focused="if ($event) { focus_label = f.name; focus_help = f.help }",
    )
    with html.Div(v_else_if="row.widget === 'records'", classes="ak-records mb-2"):
        html.Div("{{ row.label }}", classes="text-caption text-medium-emphasis mb-1")
        with v3.VSheet(
            v_for="(e, ei) in row.value", key="ei", border=True, rounded=True, classes="pa-2 mb-2"
        ):
            with html.Div(classes="d-flex flex-wrap ga-1"):
                with html.Template(v_for="f in row.fields", key="f.name"):
                    v3.VCheckbox(
                        v_if="f.type === 'boolean' && !f.optional",
                        **bound("e[f.name]", "sel_rows", _DIRTY),
                        label=("f.name",),
                        density="compact",
                        hide_details=True,
                        style="flex: 0 0 31%;",
                    )
                    v3.VSelect(
                        v_else_if="f.type === 'boolean'",
                        **bound("e[f.name]", "sel_rows", _DIRTY),
                        items=("row.items",),
                        label=("f.label",),
                        style="flex: 0 0 48%;",
                        **field,
                    )
                    v3.VTextField(
                        v_else=True,
                        **bound("e[f.name]", "sel_rows", _DIRTY),
                        label=("f.label",),
                        placeholder=("f.placeholder",),
                        persistent_placeholder=True,
                        suffix=("f.suffix",),
                        style="flex: 0 0 48%;",
                        **field,
                    )
            with html.Div(classes="d-flex justify-end"):
                move_button("row.value", "ei", -1, "sel_rows", _DIRTY)
                move_button("row.value", "ei", 1, "sel_rows", _DIRTY)
                remove_button("row.value", "ei", "sel_rows", _DIRTY)
        html.Div(
            "{{ sel_errors[row.key] }}",
            v_if="sel_errors[row.key]",
            classes="text-caption text-error mb-1",
        )
        add_button("Add a {{ row.item }}", "row.value", "{...row.blank}", "sel_rows", _DIRTY)
