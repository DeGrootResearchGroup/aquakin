"""The model panel's page: the Model tab and the dialog that names a new model.

The draft (``model_draft``) is edited in the browser and sent whole when a
field loses focus, never on each keystroke; Python's answers come back in
state of their own (``model_errors``, ``model_pointers``, ``model_uses``,
``model_offers``), so a round of checks never redraws what is being typed.
Each builder is bound onto :class:`~aquakin_ui.app.Editor` as a method
(``self`` is the editor).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from trame.widgets import html
from trame.widgets import vuetify3 as v3

from aquakin_ui.repeated import add_button, bound

if TYPE_CHECKING:
    from aquakin_ui.app import Editor

#: A field of the draft changed in the browser: Apply turns "Apply changes".
_DIRTY = "model_dirty = true"

#: A new parameter row, as the page appends it.
_NEW_PARAMETER = "{name: '', reaction: null, value: '', units: '', description: '', new: true}"

#: The panel's own errors stop Apply until they are fixed.
_HAS_ERRORS = "Object.keys(model_errors).length > 0 || model_general.length > 0"


def _field(key: str, send: str, **kwargs) -> None:
    """A text field of the draft, with aquakin's error for page key ``key`` (a
    template expression) under it, and a caret at the error's position."""
    v3.VTextField(
        error_messages=(f"model_errors[{key}] || []",),
        update_focused=send,
        density="compact",
        variant="outlined",
        hide_details="auto",
        persistent_placeholder=True,
        **kwargs,
    )
    html.Pre(
        f"{{{{ model_pointers[{key}] }}}}",
        v_if=f"model_pointers[{key}]",
        classes="text-caption text-error mb-1 ak-pointer",
    )
    html.Div(
        f"{{{{ model_uses[{key}] }}}}",
        v_if=f"model_uses[{key}]",
        classes="text-caption text-medium-emphasis mb-1",
    )


def _model_panel(self: Editor) -> None:
    # Sent when a field loses focus, and only once something was edited. (A
    # handler with no ';' is compiled as one expression, where 'if' cannot stand.)
    send = (
        f"if (!$event && model_dirty) {{ "
        f"trigger('{self.server.trigger_name(self.model_edited)}', [model_draft]); }}"
    )
    html.Div(
        "No unit uses a model yet.",
        v_if="!model_choices.length",
        classes="text-body-2 text-medium-emphasis",
    )
    with html.Div(v_else=True, classes="ak-model"):
        v3.VSelect(
            label="Model",
            items=("model_choices",),
            model_value=("model_name",),
            update_modelValue=(self.open_model, "[$event]"),
            disabled=("model_dirty || busy",),
            title="Apply or discard the edits to choose another model",
            density="compact",
            variant="outlined",
            hide_details=True,
            classes="mb-1",
        )
        html.Div("{{ model_source }}", classes="text-caption text-medium-emphasis mb-2")
        with html.Div(classes="d-flex ga-2 mb-2"):
            v3.VBtn(
                "Apply",
                click=(self.apply_model, "[model_draft, sel_dirty ? sel_rows : null]"),
                disabled=(f"!model_dirty || busy || {_HAS_ERRORS}",),
                color="primary",
                variant=("model_dirty ? 'flat' : 'tonal'",),
                classes="text-none flex-grow-1",
            )
            v3.VBtn(
                "Discard edits",
                v_if="model_dirty",
                click=self.discard_model_edits,
                variant="text",
                classes="text-none",
            )
        html.Div(
            "aquakin loaded and compiled the edited model.",
            v_if="model_valid",
            classes="text-caption text-success mb-2",
        )
        v3.VAlert(
            "{{ m }}",
            v_for="m in model_general",
            key="m",
            type="error",
            variant="tonal",
            density="compact",
            classes="mb-2 text-caption",
        )
        with html.Div(v_if="model_offers.length", classes="mb-2"):
            v3.VBtn(
                "Add parameter {{ n }}",
                v_for="n in model_offers",
                key="n",
                click=(self.add_model_parameter, "[model_draft, n]"),
                prepend_icon="mdi-plus",
                size="small",
                variant="tonal",
                color="primary",
                classes="text-none mr-1 mb-1 ak-offer",
            )
        _reactions(send)
        _parameters(send)
        with v3.VSheet(rounded=True, border=True, classes="pa-2 mt-2 text-caption"):
            html.Div("Names an expression may use", classes="font-weight-medium mb-1")
            with html.Div(v_for="(names, what) in model_names", key="what"):
                html.Span("{{ what }}: ", classes="font-weight-medium")
                html.Span("{{ names.join(', ') }}", classes="text-medium-emphasis")


def _reactions(send: str) -> None:
    html.Div("Reactions", classes="text-overline")
    with v3.VSheet(
        v_for="(r, ri) in model_draft.reactions",
        key="r.name",
        border=True,
        rounded=True,
        classes="pa-2 mb-2 ak-reaction",
    ):
        html.Div("{{ r.name }}", classes="text-body-2 font-weight-medium")
        html.Div("{{ r.description }}", classes="text-caption text-medium-emphasis mb-1")
        _field(
            "'rate.' + ri",
            send,
            label="rate",
            classes="mb-1 ak-rate",
            **bound("r.rate", "model_draft", _DIRTY),
        )
        html.Div("stoichiometry", classes="text-caption text-medium-emphasis")
        with html.Div(classes="d-flex flex-wrap ga-1"):
            with html.Div(
                v_for="(e, ei) in r.stoichiometry", key="e.species", style="flex: 0 0 48%;"
            ):
                _field(
                    "'stoich.' + ri + '.' + ei",
                    send,
                    label=("e.species",),
                    **bound("e.value", "model_draft", _DIRTY),
                )


def _parameters(send: str) -> None:
    html.Div("Parameters", classes="text-overline")
    with html.Div(v_for="(p, pi) in model_draft.parameters", key="pi", classes="mb-2 ak-parameter"):
        with html.Div(v_if="!p.new", classes="text-body-2"):
            html.Span("{{ p.name }}", classes="font-weight-medium")
            html.Span(
                " (reaction {{ p.reaction }})",
                v_if="p.reaction",
                classes="text-caption text-medium-emphasis",
            )
        _field(
            "'param.' + pi",
            send,
            v_if="p.new",
            label="name",
            classes="mb-1",
            **bound("p.name", "model_draft", _DIRTY),
        )
        with html.Div(classes="d-flex ga-1"):
            for name, flex in (("value", "0 0 34%"), ("units", "1 1 0")):
                v3.VTextField(
                    label=name,
                    density="compact",
                    variant="outlined",
                    hide_details=True,
                    update_focused=send,
                    style=f"flex: {flex};",
                    **bound(f"p.{name}", "model_draft", _DIRTY),
                )
        v3.VTextField(
            label="description",
            error_messages=("model_errors['param.' + pi] || []",),
            density="compact",
            variant="outlined",
            hide_details="auto",
            update_focused=send,
            classes="mt-1",
            **bound("p.description", "model_draft", _DIRTY),
        )
    add_button("Add a parameter", "model_draft.parameters", _NEW_PARAMETER, "model_draft", _DIRTY)


def _model_dialog(self: Editor) -> None:
    with v3.VDialog(v_model=("model_dialog_open",), max_width=520, persistent=True):
        with v3.VCard(title="Save as a new model"):
            with v3.VCardText():
                html.Div(
                    "{{ model_name }} is shipped with aquakin and is not changed. The edits "
                    "are saved as a new model of the project, which extends it.",
                    classes="mb-3",
                )
                v3.VTextField(
                    v_model=("model_new_name",),
                    label="name of the new model",
                    density="compact",
                    variant="outlined",
                    hide_details="auto",
                    error_messages=("model_dialog_error ? [model_dialog_error] : []",),
                    update_focused=(
                        f"if (!$event) {{ trigger("
                        f"'{self.server.trigger_name(self.plan_model_name)}', [model_new_name]); }}"
                    ),
                    classes="mb-2",
                )
                html.Div(
                    "These units use {{ model_name }} and will use the new model:",
                    v_if="model_dialog_units.length",
                )
                html.Div(
                    "{{ u }}",
                    v_for="u in model_dialog_units",
                    key="u",
                    classes="text-mono ak-dialog-unit",
                )
            with v3.VCardActions():
                v3.VSpacer()
                v3.VBtn(
                    "Cancel",
                    click=(self.answer_model_dialog, "['cancel']"),
                    variant="text",
                    classes="text-none",
                )
                v3.VBtn(
                    "Save new model",
                    click=(self.answer_model_dialog, "['save', model_new_name]"),
                    disabled=("busy || !model_new_name",),
                    color="primary",
                    variant="flat",
                    classes="text-none",
                )
