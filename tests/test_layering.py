"""Imports inside ember point one way, as docs/design/v0.2.md specifies."""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "ember"

# Each unit may import only the units listed for it (and itself).
ALLOWED: dict[str, set[str]] = {
    "errors": set(),
    "config": {"errors"},
    "context": {"errors", "config"},
    "xcs": {"errors", "config", "context"},
    "llm": {"errors", "config", "context", "xcs"},
    "operators": {"errors", "config", "context", "xcs", "llm"},
    "non": {"errors", "config", "context", "xcs", "llm", "operators"},
    "data": {"errors", "config", "context", "xcs", "llm", "operators"},
    "eval": {"errors", "config", "context", "xcs", "llm", "operators", "non", "data"},
    "cli": {"errors", "config", "context", "xcs", "llm", "operators", "non", "data", "eval"},
}


def _unit(path: Path) -> str:
    relative = path.relative_to(SRC)
    return relative.parts[0].removesuffix(".py")


def _imported_units(path: Path) -> set[str]:
    units: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module]
        for name in names:
            parts = name.split(".")
            if parts[0] == "ember" and len(parts) > 1:
                units.add(parts[1])
    return units


def test_imports_follow_the_layering_table() -> None:
    violations = []
    for path in sorted(SRC.rglob("*.py")):
        if path.name == "__init__.py" and path.parent == SRC:
            continue
        unit = _unit(path)
        assert unit in ALLOWED, f"{path} is in a unit the layering table does not list"
        for imported in _imported_units(path) - {unit}:
            if imported not in ALLOWED[unit]:
                violations.append(f"{path.relative_to(SRC)} imports ember.{imported}")
    assert not violations, "\n".join(violations)
