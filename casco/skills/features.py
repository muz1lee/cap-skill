"""Static feature extraction for CaSCo candidate code."""

from __future__ import annotations

import ast
from typing import Any


def _base_features(code: str) -> dict[str, Any]:
    lines = code.splitlines()
    return {
        "line_count": len(lines),
        "nonempty_line_count": sum(1 for line in lines if line.strip()),
        "function_count": 0,
        "imports": [],
        "call_names": [],
        "uses_conditionals": False,
        "uses_loops": False,
        "has_exception_handler": False,
        "parse_error": None,
    }


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def analyze_code_features(code: str) -> dict[str, Any]:
    """Return deterministic static traits for one generated code artifact."""
    features = _base_features(code)
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        features["parse_error"] = str(exc)
        return features

    imports: set[str] = set()
    call_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            features["function_count"] += 1
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name:
                call_names.add(name)
        elif isinstance(node, (ast.If, ast.IfExp)):
            features["uses_conditionals"] = True
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            features["uses_loops"] = True
        elif isinstance(node, ast.Try):
            features["has_exception_handler"] = True

    features["imports"] = sorted(imports)
    features["call_names"] = sorted(call_names)
    return features
