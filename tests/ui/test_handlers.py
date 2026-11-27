"""Every event handler the page's template holds can be compiled by Vue.

Vue compiles an inline handler that holds no ``;`` as one expression
(``$event => (<handler>)``), where a statement such as ``if`` cannot stand: the
template then fails to compile and the whole page stays blank, which no
``Editor`` test sees. A handler with a statement must therefore hold a ``;``
(``"if (!$event) { trigger(...); }"``).
"""

import html
import itertools
import re

from trame.app import get_server

from aquakin_ui.app import Editor

from .inprocess import InProcessWorker

#: An event binding in the template: ``@event="handler"`` or ``v-on:event="handler"``.
_HANDLER = re.compile(r'(?:@|v-on:)[\w:.-]+="([^"]*)"')

#: A JavaScript statement keyword, which an expression cannot hold.
_STATEMENT = re.compile(r"\b(if|for|while|do|switch|const|let|var|return|throw|try)\b")

_names = itertools.count()


def uncompilable_handlers(template: str) -> list[str]:
    """The handlers of ``template`` that hold a statement but no ``;``."""
    handlers = (html.unescape(h) for h in _HANDLER.findall(template))
    return [h for h in handlers if _STATEMENT.search(h) and ";" not in h]


def test_every_handler_the_page_builds_compiles():
    server = get_server(name=f"aquakin-ui-handlers-{next(_names)}", client_type="vue3")
    Editor(server, worker=InProcessWorker())
    template = server.state["trame__template_main"]
    assert len(_HANDLER.findall(template)) > 50  # the scan sees the page's handlers
    assert uncompilable_handlers(template) == []


def test_the_handler_that_blanked_the_page_is_caught():
    """The model panel's first send-on-blur handler, which Vue could not compile."""
    broken = "if (!$event && model_dirty) trigger('trigger__22', [model_draft])"
    fixed = "if (!$event && model_dirty) { trigger('trigger__22', [model_draft]); }"
    template = f'<v-text-field @update:focused="{broken}" v-on:blur="{fixed}" @click="x = 1" />'
    assert uncompilable_handlers(template) == [broken]
