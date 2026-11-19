"""The interface never imports aquakin or JAX, and aquakin never imports the interface."""

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_importing_the_interface_loads_neither_aquakin_nor_jax():
    code = (
        "import sys\n"
        "import aquakin_ui, aquakin_ui.app, aquakin_ui.catalog, aquakin_ui.document\n"
        "import aquakin_ui.forms, aquakin_ui.icons, aquakin_ui.worker, aquakin_ui.widget\n"
        "bad = [m for m in sys.modules if m in ('jax', 'aquakin') or m.startswith(('jax.', 'aquakin.'))]\n"
        "print(','.join(sorted(bad)))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""


def test_aquakin_never_imports_the_interface():
    for path in (ROOT / "aquakin").rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            assert not any(n.split(".")[0] == "aquakin_ui" for n in names), path
