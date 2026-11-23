"""The keyword arguments that messages and docstrings tell users to pass exist.

An error message that says "pass ``gradient='stable_adjoint'``" is text: when
the keyword is renamed nothing points from the signature back to the message,
so the advice goes stale silently, and a user who follows it gets a
``TypeError`` on top of the error they were trying to fix. This module checks
the advice against the real signatures, in two shapes:

- **a call**, ``name(kw=...)``, where ``name`` is a public callable
  (``dgsm``, ``aquakin.calibrate``, ``Plant.solve``, ``plant.calibrate``):
  every keyword in it must be a parameter of that callable, unless it takes
  ``**kwargs``. A lower-case receiver (``plant.``, ``reactor.``) stands for the
  public classes whose name ends in it, and the keyword must be a parameter of
  that method on at least one of them. Names that are not public callables of
  ``aquakin`` (``jax.jacfwd``, private helpers) are not checked.
- **a bare keyword**, ``kw=value``, in the text of a ``raise`` or a
  ``warnings.warn`` (and in the module-level message constants they are built
  from): ``kw`` must be a parameter or field of some public callable. A message
  has no call to anchor it, so this is the weaker check, but it is what catches
  a keyword that was removed outright.

The checked text is the message strings described above, anywhere under
``aquakin/``, and the docstrings of the public API (``aquakin``,
``aquakin.plant``, ``aquakin.study`` and the methods of their public classes),
which are what ``help()`` and the API reference show. Only the call shape is
checked in docstrings: their prose names the callable it means.

What this cannot catch is advice whose keywords all exist but which does not
work (``mode="forward"`` without ``method="through_solve"`` for a reactor);
that needs a test that follows the advice (``.claude/rules/tests.md``).
"""

import ast
import dataclasses
import inspect
import re
from pathlib import Path

import aquakin
import aquakin.plant
import aquakin.study

ROOT = Path(__file__).parents[2]
PUBLIC_MODULES = (aquakin, aquakin.plant, aquakin.study)

# ``name(args)`` with an optional ``aquakin.`` prefix and at most one dot in
# the name; ``args`` holds no parentheses, so a nested call is read innermost.
CALL = re.compile(r"(?<![\w.])(?:aquakin\.)?([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)\(([^()]*)\)")
# A keyword inside a call's arguments.
CALL_KEYWORD = re.compile(r"(?<![\w.])([a-z_]\w*)=(?!=)")
# A bare ``kw=value``: the value must follow in the same string, so an
# f-string's ``m={m}`` (which ends a constant before the substitution) and a
# comparison (``==``) are not read as advice.
BARE_KEYWORD = re.compile(r"(?<![\w.\[])([a-z_][a-z0-9_]*)=(?=['\"\w\[(])")
# Keyword arguments that carry user-facing text to a helper that raises it
# (``check_finite_gradient(..., remedy=...)``).
MESSAGE_KEYWORDS = {"remedy", "message", "msg", "hint"}


def public_objects(modules=PUBLIC_MODULES):
    """
    The public callables of ``modules``, by the names a message may use.

    Returns
    -------
    dict of str to object
        Each public function and class under its own name, and each public
        method (and ``__init__``-taking class) as ``Class.method``.
    """
    found = {}
    for module in modules:
        for name in module.__all__:
            obj = getattr(module, name)
            if not callable(obj):
                continue
            found[name] = obj
            if inspect.isclass(obj):
                for attr, member in vars(obj).items():
                    if attr.startswith("_"):
                        continue
                    member = getattr(member, "__func__", member)
                    if inspect.isfunction(member):
                        found[f"{name}.{attr}"] = member
    return found


def parameters(obj):
    """
    The keyword names ``obj`` accepts, and whether it takes ``**kwargs``.

    A dataclass contributes its fields; an object without an inspectable
    signature accepts nothing that can be checked.

    Returns
    -------
    tuple of (set of str, bool)
    """
    names, open_ended = set(), False
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        return names, True
    for p in signature.parameters.values():
        if p.kind is p.VAR_KEYWORD:
            open_ended = True
        elif p.kind is not p.VAR_POSITIONAL:
            names.add(p.name)
    if dataclasses.is_dataclass(obj):
        names |= {f.name for f in dataclasses.fields(obj)}
    return names - {"self", "cls"}, open_ended


def resolve(name, objects):
    """
    The public callables a call's ``name`` can mean, or an empty list.

    ``Class.method`` and bare names are looked up directly; a lower-case
    receiver (``plant.solve``) means that method on every public class whose
    name ends in the receiver, ignoring case (``Plant``, ``BatchReactor``).
    """
    if name in objects:
        return [objects[name]]
    receiver, dot, method = name.partition(".")
    if not dot or not receiver.islower():
        return []
    return [
        obj
        for key, obj in objects.items()
        if key.endswith(f".{method}") and key.split(".")[0].lower().endswith(receiver)
    ]


def call_problems(text, objects):
    """Messages for each ``name(kw=...)`` in ``text`` whose ``kw`` is not accepted."""
    problems = []
    for name, args in CALL.findall(text):
        targets = resolve(name, objects)
        if not targets:
            continue
        accepted, open_ended = set(), False
        for target in targets:
            names, more = parameters(target)
            accepted |= names
            open_ended |= more
        if open_ended:
            continue
        for keyword in CALL_KEYWORD.findall(args):
            if keyword not in accepted:
                problems.append(f"`{name}(...)` takes no `{keyword}=`")
    return problems


def bare_problems(text, vocabulary):
    """Messages for each bare ``kw=value`` in ``text`` that no public callable takes."""
    return [
        f"no public callable takes `{keyword}=`"
        for keyword in BARE_KEYWORD.findall(text)
        if keyword not in vocabulary
    ]


def is_warn_call(node):
    """Whether ``node`` is a ``warn(...)`` / ``warnings.warn(...)`` call."""
    func = node.func if isinstance(node, ast.Call) else None
    return getattr(func, "attr", getattr(func, "id", None)) == "warn"


def message_strings(source):
    """
    The user-facing message strings of a module, with their line numbers.

    The string constants inside a ``raise``, a ``warn(...)`` call or a keyword
    argument in ``MESSAGE_KEYWORDS``; those assigned to a name any of these
    use (a message built in a local ``remedy = (...)`` first); and those
    assigned to a module-level UPPER_CASE name (the constants messages are
    built from, wherever they are used). Comments are not strings, so they are
    never read.

    Returns
    -------
    list of tuple of (int, str)
    """
    tree = ast.parse(source)
    roots = [n.exc for n in ast.walk(tree) if isinstance(n, ast.Raise) and n.exc is not None]
    roots += [n for n in ast.walk(tree) if is_warn_call(n)]
    roots += [
        k.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        for k in n.keywords
        if k.arg in MESSAGE_KEYWORDS
    ]
    used = {n.id for root in roots for n in ast.walk(root) if isinstance(n, ast.Name)}
    roots += [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in used for t in n.targets)
    ]
    roots += [
        n.value
        for n in tree.body
        if isinstance(n, ast.Assign)
        and all(isinstance(t, ast.Name) and t.id.isupper() for t in n.targets)
    ]
    return sorted(
        {
            (c.lineno, c.value)
            for root in roots
            for c in ast.walk(root)
            if isinstance(c, ast.Constant) and isinstance(c.value, str)
        }
    )


def hint_problems(package_dir, objects):
    """
    Every stale keyword hint in the messages under ``package_dir`` and in the
    docstrings of ``objects``.

    Returns
    -------
    list of str
        One line per problem, naming where it is and what is wrong.
    """
    vocabulary = set()
    for obj in objects.values():
        vocabulary |= parameters(obj)[0]
    problems = []
    for file in sorted(package_dir.rglob("*.py")):
        where = file.relative_to(package_dir.parent).as_posix()
        for lineno, text in message_strings(file.read_text()):
            # A keyword inside a call that resolves is the call check's to
            # judge; reading it as a bare keyword too would report it twice.
            loose = CALL.sub(lambda m: "" if resolve(m.group(1), objects) else m.group(0), text)
            for why in call_problems(text, objects) + bare_problems(loose, vocabulary):
                problems.append(f"{where}:{lineno}: {why}")
    for name, obj in sorted(objects.items()):
        for why in call_problems(inspect.getdoc(obj) or "", objects):
            problems.append(f"docstring of {name}: {why}")
    return sorted(set(problems))


ADVICE = (
    "A message or docstring tells users to pass a keyword the API does not "
    "take. Name the current keyword (and check the advice works: "
    ".claude/rules/tests.md asks for a test that follows it)."
)


# --- the check on this package ------------------------------------------------


def test_keyword_hints_name_existing_parameters():
    problems = hint_problems(ROOT / "aquakin", public_objects())
    assert not problems, "\n".join(problems) + "\n" + ADVICE


# --- the check itself, on small synthetic inputs ------------------------------


class Plant:
    def solve(self, t_span, *, integrator=None, diff=None):
        """Solve; pass ``integrator=`` and ``diff=``."""

    def calibrate(self, obs, *, progress=None, **options):
        """Calibrate."""


class BatchReactor:
    def solve(self, t_span, *, rtol=1e-6):
        """Solve."""


@dataclasses.dataclass
class DifferentiationConfig:
    mode: str = "reverse"
    method: str = "stable"


def dgsm(fn, ranges, *, diff=None):
    """Use ``dgsm(fn, ranges, diff=DifferentiationConfig(mode="forward"))``."""


OBJECTS = {
    "Plant": Plant,
    "Plant.solve": Plant.solve,
    "Plant.calibrate": Plant.calibrate,
    "BatchReactor": BatchReactor,
    "BatchReactor.solve": BatchReactor.solve,
    "DifferentiationConfig": DifferentiationConfig,
    "dgsm": dgsm,
}
VOCABULARY = {"t_span", "integrator", "diff", "obs", "progress", "rtol", "mode", "method"}


def test_calls_are_checked_against_their_own_signature():
    good = (
        "Pass aquakin.dgsm(fn, r, diff=DifferentiationConfig(mode='forward', "
        "method='through_solve')), Plant.solve(t, integrator=x), "
        "plant.calibrate(o, anything=1), reactor.solve(t, rtol=1e-8), "
        "jax.jacfwd(f, argnums=0), _private(x=1)."
    )
    assert call_problems(good, OBJECTS) == []
    bad = (
        "Pass dgsm(fn, r, ad_mode='forward'), Plant.solve(t, solver=x), "
        "DifferentiationConfig(gradient='stable'), reactor.solve(t, adjoint=a)."
    )
    assert call_problems(bad, OBJECTS) == [
        "`dgsm(...)` takes no `ad_mode=`",
        "`Plant.solve(...)` takes no `solver=`",
        "`DifferentiationConfig(...)` takes no `gradient=`",
        "`reactor.solve(...)` takes no `adjoint=`",
    ]


def test_bare_keywords_need_a_value_and_a_public_owner():
    assert bare_problems("use diff=X or mode='forward'; got m=", VOCABULARY) == []
    assert bare_problems("a == b, x[k]=1", VOCABULARY) == []
    assert bare_problems("pass gradient='stable_adjoint' or ad_mode=1", VOCABULARY) == [
        "no public callable takes `gradient=`",
        "no public callable takes `ad_mode=`",
    ]


def test_messages_come_from_raise_warn_and_message_constants():
    source = '''
"""Module docstring with gradient='x' is not a message."""
import warnings
REMEDY = "pass ad_mode='forward'"
lower = "pass a=1"


def f(m):
    # a comment with gradient='x' is not a string
    warnings.warn("use stale='a'")
    raise ValueError(f"got m={m}; pass " + REMEDY + " or old='b'")


def g(ok):
    remedy = "pass local='c'" if ok else "pass other='d'"
    unrelated = "not a message, e='f'"
    check(probe, remedy=remedy, what="label g='h'")
'''
    assert message_strings(source) == [
        (4, "pass ad_mode='forward'"),
        (10, "use stale='a'"),
        (11, " or old='b'"),
        (11, "; pass "),
        (11, "got m="),
        (15, "pass local='c'"),
        (15, "pass other='d'"),
    ]


def test_hint_problems_report_files_and_docstrings(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "m.py").write_text(
        'MSG = "Plant.solve(t, solver=1)"\n'
        "def f():\n"
        "    raise ValueError(\"pass gradient='stable_adjoint'\")\n"
    )

    def documented():
        """Call ``Plant.solve(t, gradient='auto')``."""

    objects = {**OBJECTS, "documented": documented}
    assert hint_problems(package, objects) == [
        "docstring of documented: `Plant.solve(...)` takes no `gradient=`",
        "pkg/m.py:1: `Plant.solve(...)` takes no `solver=`",
        "pkg/m.py:3: no public callable takes `gradient=`",
    ]


def test_public_objects_cover_functions_classes_and_methods():
    objects = public_objects()
    assert objects["dgsm"] is aquakin.dgsm
    assert objects["Plant.solve"] is aquakin.Plant.solve
    assert "DifferentiationConfig" in objects
    assert not any(name.split(".")[-1].startswith("_") for name in objects)
