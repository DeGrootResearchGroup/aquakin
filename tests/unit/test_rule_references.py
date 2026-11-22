"""The Claude-facing files name only code and files that exist.

``CLAUDE.md`` says a rule that names a function that no longer exists is worse
than no rule ("Maintaining the Claude-facing files"): the reader trusts it, goes
looking, and either wastes the time or recreates the thing under the old name.
Rules are changed in the same commit as the code they describe, but a rename in
one module routinely misses the rule file that mentions it. This module checks
the inline-code spans (single backticks) of ``CLAUDE.md`` and every rule file
under ``.claude/rules/`` for two shapes of reference whose target can be
verified without guessing:

- **a repository path**: a span containing ``/`` whose first segment is a
  top-level directory of the repository or a subpackage of ``aquakin`` (rules
  write ``plant/units.py`` for ``aquakin/plant/units.py``). It must exist. A
  path ending in ``.SYMBOL`` (``core/precedence.ASSOCIATIVE_CALLS``) names a
  module and a name defined in it, and both must exist. Spans with glob or
  placeholder characters (``*``, ``{``, ``<``, a space) are patterns, not paths,
  and are skipped.
- **a pytest node id**: ``file.py::name``, with or without a directory
  (``test_mass_balance_plant.py::test_...``). The file must exist (a bare file
  name anywhere under the Python trees below) and define ``name``.
- **a Python name of this repository**: an identifier, optionally dotted and
  optionally followed by ``()``, that is either private (a segment starting
  with one underscore, which a third-party name in a rule almost never is) or
  dotted from a class or module of ``aquakin`` / ``aquakin_ui``
  (``Plant.solve_sensitivity``, ``kernels.safe_div``). Its last segment must be
  defined somewhere in the repository's Python: a function, class, parameter,
  keyword argument, or assigned name or attribute.

Bare public names (``make_cstr_plant``) and dotted names from other libraries
(``jax.lax.custom_root``) are not checked: a bare name cannot be told apart
from a YAML key, an expression-language function or a JavaScript identifier,
and checking them would fail on correct rules. Fenced code blocks are skipped.

``.claude/notes/`` is not checked: notes are dated logs, and a log may name
what has since been removed. Everything here is text and AST processing: no
JAX, no ``aquakin`` import.
"""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).parents[2]
PACKAGES = ("aquakin", "aquakin_ui")
# Where a defined name may live. ``tests`` and ``scripts`` are included because
# rules name fixtures and helpers from them.
PYTHON_TREES = ("aquakin", "aquakin_ui", "tests", "scripts")
NOT_A_PATH = set("*{}<> ")

SPAN = re.compile(r"`([^`\n]+)`")
FENCE = re.compile(r"^[ \t]*(```|~~~).*?^[ \t]*\1[ \t]*$", re.DOTALL | re.MULTILINE)
IDENTIFIER = re.compile(r"([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)(?:\(\))?")
NODE_ID = re.compile(r"([\w./-]+\.py)::(\w+)")


def checked_files(root):
    """``CLAUDE.md`` and every rule file, as repository-relative paths."""
    rules = sorted((root / ".claude" / "rules").rglob("*.md"))
    return [root / "CLAUDE.md", *rules]


def code_spans(text):
    """The inline-code spans of Markdown ``text``, outside fenced code blocks."""
    return SPAN.findall(FENCE.sub("", text))


def defined_names(source):
    """
    Every name a Python module defines or binds.

    Functions, classes, parameters, keyword-argument names at call sites, and
    names and attributes assigned to (including ``for`` targets, ``with ... as``
    and imports' bound names). A rule may name any of these: a method bound onto
    a class by assignment, a dataclass field, a keyword option.

    Parameters
    ----------
    source : str
        The module's source text.

    Returns
    -------
    set of str
    """
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            names.add(node.attr)
        elif isinstance(node, ast.alias):
            names.add((node.asname or node.name).split(".")[-1])
    return names


class Repository:
    """The names, classes, modules and top-level directories of a checkout."""

    def __init__(self, root):
        self.root = root
        self.defined = set()
        self.classes = set()
        self.modules = set(PACKAGES)
        for tree in PYTHON_TREES:
            for file in sorted((root / tree).rglob("*.py")):
                source = file.read_text()
                self.defined |= defined_names(source)
                if tree in PACKAGES:
                    self.modules.add(file.parent.name if file.stem == "__init__" else file.stem)
                    self.classes |= {
                        n.name for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ClassDef)
                    }
        self.top = {p.name for p in root.iterdir() if p.is_dir()}
        self.subpackages = {
            p.name for p in (root / "aquakin").iterdir() if (p / "__init__.py").is_file()
        }

    def resolve(self, path):
        """The file or directory ``path`` names, or None."""
        for base in (self.root, self.root / "aquakin"):
            if (base / path).exists():
                return base / path
        return None

    def path_problem(self, span):
        """Why a path-shaped ``span`` names nothing, or None when it is fine."""
        path = span.rstrip("/")
        if self.resolve(path) is not None:
            return None
        module, dot, symbol = path.rpartition(".")
        if dot and symbol.isidentifier():
            file = self.resolve(module + ".py")
            if file is not None:
                if symbol in defined_names(file.read_text()):
                    return None
                return f"{module}.py defines no `{symbol}`"
        return "no such file or directory"

    def node_problem(self, file, name):
        """Why a ``file.py::name`` node id names nothing, or None when it is fine."""
        if "/" in file:
            found = self.resolve(file)
            candidates = [found] if found is not None else []
        else:
            candidates = [
                f for tree in PYTHON_TREES for f in sorted((self.root / tree).rglob(file))
            ]
        if not candidates:
            return "no such file"
        if any(name in defined_names(f.read_text()) for f in candidates):
            return None
        return f"{file} defines no `{name}`"

    def name_problem(self, parts):
        """Why a checked dotted name names nothing, or None when it is fine."""
        last = parts[-1]
        if last in self.defined or last in self.modules:
            return None
        return f"nothing in {', '.join(PYTHON_TREES)} defines `{last}`"

    def is_path(self, span):
        """Whether ``span`` is a repository path this module checks."""
        if "/" not in span or NOT_A_PATH & set(span):
            return False
        first = span.split("/")[0]
        return first in self.top or first in self.subpackages

    def is_checked_name(self, parts):
        """Whether a dotted name is private or rooted at this repository."""
        private = any(p.startswith("_") and not p.startswith("__") for p in parts)
        rooted = len(parts) > 1 and (parts[0] in self.classes or parts[0] in self.modules)
        return private or rooted


def reference_problems(root):
    """
    Every path or repository name in the Claude-facing files that is missing.

    Returns
    -------
    list of str
        One message per distinct missing reference per file, naming the file,
        the span and what is missing.
    """
    repo = Repository(root)
    problems = []
    for file in checked_files(root):
        rel = file.relative_to(root).as_posix()
        for span in sorted(set(code_spans(file.read_text()))):
            span = span.strip()
            node = NODE_ID.fullmatch(span)
            if node:
                why = repo.node_problem(*node.groups())
            elif repo.is_path(span):
                why = repo.path_problem(span)
            else:
                match = IDENTIFIER.fullmatch(span)
                parts = match.group(1).split(".") if match else None
                why = repo.name_problem(parts) if parts and repo.is_checked_name(parts) else None
            if why:
                problems.append(f"{rel}: `{span}`: {why}")
    return problems


ADVICE = (
    "Update the rule to the current name or path, or delete the sentence if what "
    "it described is gone. A rule that names something that no longer exists is "
    "worse than no rule (CLAUDE.md, 'Maintaining the Claude-facing files')."
)


# --- the check on this repository --------------------------------------------


def test_claude_facing_files_name_only_existing_code():
    problems = reference_problems(ROOT)
    assert not problems, "\n".join(problems) + "\n" + ADVICE


# --- the check itself, on a small synthetic tree ------------------------------

MODULE = '''
class Plant:
    def solve(self, t_end, *, dtmax=None):
        self.cache = {}

def _private_helper():
    pass

ASSOCIATIVE_CALLS = ("min", "max")
'''


def _tree(tmp_path, rule):
    (tmp_path / "aquakin" / "plant").mkdir(parents=True)
    (tmp_path / "aquakin" / "__init__.py").write_text("")
    (tmp_path / "aquakin" / "plant" / "__init__.py").write_text("")
    (tmp_path / "aquakin" / "plant" / "plant.py").write_text(MODULE)
    for tree in PYTHON_TREES[1:]:
        (tmp_path / tree).mkdir()
    (tmp_path / ".claude" / "rules").mkdir(parents=True)
    (tmp_path / ".claude" / "rules" / "r.md").write_text(rule)
    (tmp_path / "CLAUDE.md").write_text("# Briefing\n")
    return tmp_path


def test_existing_references_pass(tmp_path):
    rule = (
        "Use `plant/plant.py`, `aquakin/plant/` and `plant/plant.ASSOCIATIVE_CALLS`; "
        "call `Plant.solve()` with `dtmax`, `Plant.cache`, `_private_helper`, "
        "`plant.Plant`, `plant/plant.py::_private_helper`, `plant.py::Plant`. Not checked: `make_thing`, `jax.lax.custom_root`, "
        "`tests/**`, `plant/<unit>.py`, `max(a, b)`, `paths:`.\n"
    )
    assert reference_problems(_tree(tmp_path, rule)) == []


def test_missing_references_are_reported(tmp_path):
    rule = (
        "See `plant/gone.py`, `plant/plant.NOT_THERE`, `Plant.removed_method()`, "
        "`_renamed_helper`, `_Problem.field`, `plant/plant.py::gone`, "
        "`test_gone.py::test_x`.\n"
    )
    assert reference_problems(_tree(tmp_path, rule)) == [
        ".claude/rules/r.md: `Plant.removed_method()`: nothing in aquakin, "
        "aquakin_ui, tests, scripts defines `removed_method`",
        ".claude/rules/r.md: `_Problem.field`: nothing in aquakin, aquakin_ui, "
        "tests, scripts defines `field`",
        ".claude/rules/r.md: `_renamed_helper`: nothing in aquakin, aquakin_ui, "
        "tests, scripts defines `_renamed_helper`",
        ".claude/rules/r.md: `plant/gone.py`: no such file or directory",
        ".claude/rules/r.md: `plant/plant.NOT_THERE`: plant/plant.py defines no `NOT_THERE`",
        ".claude/rules/r.md: `plant/plant.py::gone`: plant/plant.py defines no `gone`",
        ".claude/rules/r.md: `test_gone.py::test_x`: no such file",
    ]


def test_fenced_code_blocks_are_not_checked(tmp_path):
    rule = "```bash\npytest `plant/gone.py`\n```\n\nThen `plant/also_gone.py`.\n"
    [msg] = reference_problems(_tree(tmp_path, rule))
    assert "`plant/also_gone.py`" in msg
