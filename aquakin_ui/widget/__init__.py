"""A trame widget wrapping Vue Flow as a plant flowsheet canvas.

The JavaScript source is in the repository's ``ui-widget/`` folder; ``npm run
build`` there writes the bundle into ``serve/``, which ships with the package,
so using the widget needs no JavaScript toolchain.
"""

from trame_client.widgets.core import AbstractElement

from aquakin_ui.widget import module


class Flowsheet(AbstractElement):
    """The flowsheet canvas.

    Parameters
    ----------
    nodes, edges : str or tuple
        State bindings for the drawn units and connections.
    drag_type : str, optional
        The drag-and-drop data type a palette item carries.
    fit_key : str or tuple, optional
        A state binding; changing its value frames the whole flowsheet again.
    icons : str or tuple, optional
        A state binding to ``{icon name: SVG markup}``; nodes name their icon.
    connect, move, remove, dropped, select : callable or str
        Handlers for finished edits; each receives one event payload.
    """

    # trame serves these through AbstractElement.__getattr__ / __setattr__ (as
    # aliases of its props and events lists); declared so their type is known.
    _attr_names: list
    _event_names: list

    def __init__(self, children=None, **kwargs):
        super().__init__("aquakin-flowsheet", children, **kwargs)
        if self.server:
            self.server.enable_module(module)
        self._attr_names += [
            "nodes",
            "edges",
            "icons",
            ("drag_type", "dragType"),
            ("fit_key", "fitKey"),
        ]
        self._event_names += ["connect", "move", "remove", "dropped", "select"]
