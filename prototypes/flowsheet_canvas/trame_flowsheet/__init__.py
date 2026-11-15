"""A trame widget wrapping Vue Flow as a plant flowsheet canvas.

The JavaScript is built once (``js/``, ``npm run build``) into ``serve/`` and
shipped with the package, so using the widget needs no JavaScript toolchain.
"""

from trame_client.widgets.core import AbstractElement

from trame_flowsheet import module


class Flowsheet(AbstractElement):
    """The flowsheet canvas.

    Parameters
    ----------
    nodes, edges : str or tuple
        State bindings for the drawn units and connections.
    connect, move, remove, dropped, select : callable or str
        Handlers for finished edits; each receives one event payload.
    """

    def __init__(self, children=None, **kwargs):
        super().__init__("aquakin-flowsheet", children, **kwargs)
        if self.server:
            self.server.enable_module(module)
        self._attr_names += ["nodes", "edges", ("drag_type", "dragType")]
        self._event_names += ["connect", "move", "remove", "dropped", "select"]
