"""Private ``Plant`` members used outside ``plant.py`` are listed in ``aquakin.plant.internals``.

``SHARED_INTERNALS`` names, per module, the private members of ``Plant`` that
module reaches on a plant object. These checks keep the list complete (an
unlisted use fails) and current (a listed name that is no longer a ``Plant``
member, or that its module no longer uses, fails).

A use is an attribute access ``<expr>._name`` whose ``_name`` is a private
``Plant`` member and whose receiver is not the bare name ``self``: inside a
module bound onto ``Plant`` (``dynamic_solve``, ``steady_solve``) ``self`` is
the plant itself, and in any other class ``self._name`` is that class's own
attribute.
"""

import ast
from collections import defaultdict
from pathlib import Path

from aquakin.plant.internals import SHARED_INTERNALS

PACKAGE = Path(__file__).parents[2] / "aquakin"

PLANT_MODULE = "plant/plant.py"


def _private_plant_members(package: Path) -> set[str]:
    """The private names ``Plant`` defines: its methods, class-body
    assignments (including the bound module functions) and the attributes its
    methods store on ``self``."""
    tree = ast.parse((package / PLANT_MODULE).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Plant")
    members = set()
    for node in cls.body:
        if isinstance(node, ast.FunctionDef):
            members.add(node.name)
        elif isinstance(node, ast.Assign):
            members.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            members.add(node.target.id)
    for node in ast.walk(cls):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Store)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            members.add(node.attr)
    return {m for m in members if m.startswith("_") and not m.endswith("__")}


def _uses(package: Path) -> dict[str, set[str]]:
    """``{module: names}`` of the private ``Plant`` members each module other
    than ``plant.py`` reaches through a receiver that is not ``self``."""
    private = _private_plant_members(package)
    found = defaultdict(set)
    for path in sorted(package.rglob("*.py")):
        rel = path.relative_to(package).as_posix()
        if rel == PLANT_MODULE:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Attribute)
                and node.attr in private
                and not (isinstance(node.value, ast.Name) and node.value.id == "self")
            ):
                found[rel].add(node.attr)
    return dict(found)


def test_every_use_of_a_private_plant_member_is_listed():
    """Each module uses only the private ``Plant`` members listed for it."""
    unlisted = {
        rel: sorted(names - SHARED_INTERNALS.get(rel, frozenset()))
        for rel, names in _uses(PACKAGE).items()
    }
    unlisted = {rel: names for rel, names in unlisted.items() if names}
    assert unlisted == {}, "add these to aquakin/plant/internals.py:\n" + "\n".join(
        f"{rel}: {', '.join(names)}" for rel, names in sorted(unlisted.items())
    )


def test_every_listed_name_is_a_plant_member_its_module_uses():
    """No listed name is stale: each is still a private ``Plant`` member, and
    the module it is listed for still uses it."""
    members = _private_plant_members(PACKAGE)
    uses = _uses(PACKAGE)
    stale = {
        rel: sorted(n for n in names if n not in members or n not in uses.get(rel, set()))
        for rel, names in SHARED_INTERNALS.items()
    }
    stale = {rel: names for rel, names in stale.items() if names}
    assert stale == {}, "remove these from aquakin/plant/internals.py:\n" + "\n".join(
        f"{rel}: {', '.join(names)}" for rel, names in sorted(stale.items())
    )


def test_the_scan_sees_uses_and_skips_self(tmp_path):
    """The scan finds members the class stores on ``self`` and its methods,
    counts a use through ``plant`` or ``self.plant`` (also inside a nested
    function), and skips ``self._name`` and public names (so the first test
    can fail)."""
    (tmp_path / "plant").mkdir()
    (tmp_path / PLANT_MODULE).write_text(
        "class Plant:\n"
        "    def __init__(self):\n"
        "        self._cache = {}\n"
        "        self.public = 1\n"
        "    def _rhs(self):\n"
        "        return self._cache\n"
    )
    (tmp_path / "plant" / "probe.py").write_text(
        "class R:\n"
        "    def f(self):\n"
        "        def g():\n"
        "            return self.plant._rhs()\n"
        "        return self._cache, self.plant.public\n"
        "def h(plant):\n"
        "    return plant._cache\n"
    )
    assert _private_plant_members(tmp_path) == {"_cache", "_rhs"}
    assert _uses(tmp_path) == {"plant/probe.py": {"_rhs", "_cache"}}
