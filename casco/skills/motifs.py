"""Deterministic motif extraction for generated CaSCo candidate programs."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from typing import Any

ROBOT_PRIMITIVES = {
    "close_gripper",
    "get_all_object_poses",
    "get_object_pose",
    "goto_pose",
    "goto_pose_interactive_cartesian",
    "open_gripper",
    "sample_grasp_pose",
}


@dataclass(frozen=True)
class _CallEvent:
    name: str
    node: ast.Call
    lineno: int
    col_offset: int


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


class _PrimitiveCallVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.events: list[_CallEvent] = []

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        name = _call_name(node.func)
        if name and name.rsplit(".", 1)[-1] in ROBOT_PRIMITIVES:
            self.events.append(
                _CallEvent(
                    name=name.rsplit(".", 1)[-1],
                    node=node,
                    lineno=getattr(node, "lineno", 0),
                    col_offset=getattr(node, "col_offset", 0),
                )
            )
        self.generic_visit(node)


def _literal_number(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _literal_number(node.operand)
        return -value if value is not None else None
    return None


def _literal_sequence(node: ast.AST) -> list[float] | None:
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [_literal_number(item) for item in node.elts]
        return values if all(value is not None for value in values) else None
    if isinstance(node, ast.Call) and _call_name(node.func) in {"array", "np.array", "numpy.array"}:
        if node.args:
            return _literal_sequence(node.args[0])
    return None


def _contains_down_quat(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        sequence = _literal_sequence(node)
        if sequence == [0.0, 0.0, 1.0, 0.0]:
            return True
    return False


def _positive_z_offsets(tree: ast.AST) -> list[float]:
    offsets: list[float] = []
    for node in ast.walk(tree):
        sequence = _literal_sequence(node)
        if sequence and len(sequence) >= 3 and sequence[0] == 0 and sequence[1] == 0:
            if sequence[2] > 0:
                offsets.append(sequence[2])
    return offsets


def _z_offset_bucket(offsets: list[float]) -> str:
    if not offsets:
        return "none"
    largest = max(offsets)
    if largest <= 0.07:
        return "low"
    if largest <= 0.10:
        return "medium"
    return "high"


def _string_literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _uses_explicit_target_pose(events: list[_CallEvent]) -> bool:
    object_pose_literals: list[str] = []
    for event in events:
        if event.name != "get_object_pose" or not event.node.args:
            continue
        value = _string_literal(event.node.args[0])
        if value is not None:
            object_pose_literals.append(value)
    if len(object_pose_literals) >= 2:
        return True
    return any("bowl" not in value.lower() for value in object_pose_literals)


def _has_loop(tree: ast.AST) -> bool:
    return any(isinstance(node, (ast.For, ast.AsyncFor, ast.While)) for node in ast.walk(tree))


def _object_lookup_style(tree: ast.AST, events: list[_CallEvent]) -> str:
    names = [event.name for event in events]
    if "get_object_pose" in names:
        return "explicit"
    if "get_all_object_poses" in names and _has_loop(tree):
        return "loop_filter"
    if "get_all_object_poses" in names:
        return "bulk"
    return "none"


def _assigned_lift_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        offsets = _positive_z_offsets(node.value)
        if not offsets:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and ("lift" in target.id or max(offsets) >= 0.08):
                names.add(target.id)
    return names


def _contains_name(node: ast.AST, names: set[str]) -> bool:
    return any(isinstance(child, ast.Name) and child.id in names for child in ast.walk(node))


def _event_indices(events: list[_CallEvent], name: str) -> list[int]:
    return [index for index, event in enumerate(events) if event.name == name]


def _has_lift_after_grasp(tree: ast.AST, events: list[_CallEvent]) -> bool:
    close_indices = _event_indices(events, "close_gripper")
    if not close_indices:
        return False
    first_close = close_indices[0]
    first_open = next(
        (index for index, event in enumerate(events[first_close + 1 :], start=first_close + 1)
         if event.name == "open_gripper"),
        len(events),
    )
    lift_names = _assigned_lift_names(tree)
    for event in events[first_close + 1 : first_open]:
        if event.name not in {"goto_pose", "goto_pose_interactive_cartesian"}:
            continue
        if event.node.args and _contains_name(event.node.args[0], lift_names):
            return True
        if event.node.args and any(offset >= 0.08 for offset in _positive_z_offsets(event.node.args[0])):
            return True
    return False


def _orientation_constant(tree: ast.AST, events: list[_CallEvent]) -> str:
    if _contains_down_quat(tree):
        return "down_quat"
    close_indices = _event_indices(events, "close_gripper")
    if close_indices:
        after_close = events[close_indices[0] + 1 :]
        for event in after_close:
            if event.name.startswith("goto_pose") and len(event.node.args) >= 2:
                if _contains_name(event.node.args[1], {"grasp_quat", "bowl_grasp_quat"}):
                    return "grasp_quat"
    return "unknown"


def _subgoal_signature(
    events: list[_CallEvent],
    *,
    has_lift_after_grasp: bool,
    uses_explicit_target_pose: bool,
) -> list[str]:
    names = [event.name for event in events]
    signature: list[str] = []
    if "get_all_object_poses" in names or "get_object_pose" in names:
        signature.append("lookup")
    if "sample_grasp_pose" in names:
        signature.append("sample_grasp")

    close_indices = _event_indices(events, "close_gripper")
    first_close = close_indices[0] if close_indices else len(events)
    if any(event.name.startswith("goto_pose") for event in events[:first_close]):
        signature.append("approach")
    if close_indices:
        signature.append("grasp")
    if has_lift_after_grasp:
        signature.append("lift")
    if uses_explicit_target_pose:
        signature.append("target_pose")

    open_indices = _event_indices(events, "open_gripper")
    first_open = open_indices[0] if open_indices else len(events)
    goto_between_close_and_open = [
        event for event in events[first_close + 1 : first_open] if event.name.startswith("goto_pose")
    ]
    if goto_between_close_and_open and (len(goto_between_close_and_open) > 1 or not has_lift_after_grasp):
        signature.append("place")
    if open_indices:
        signature.append("release")
    if any(event.name.startswith("goto_pose") for event in events[first_open + 1 :]):
        signature.append("retreat")
    return signature


def _motif_id(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "m_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


def extract_code_motif(code: str) -> dict[str, Any]:
    """Return a deterministic structural motif for one generated program."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {
            "motif_id": "invalid_code",
            "parse_error": str(exc),
            "ordered_call_skeleton": [],
            "subgoal_signature": [],
            "has_lift_after_grasp": False,
            "uses_explicit_target_pose": False,
            "orientation_constant": "unknown",
            "z_offset_bucket": "none",
            "object_lookup_style": "none",
        }

    visitor = _PrimitiveCallVisitor()
    visitor.visit(tree)
    events = sorted(visitor.events, key=lambda event: (event.lineno, event.col_offset))
    ordered_call_skeleton = [event.name for event in events]
    has_lift = _has_lift_after_grasp(tree, events)
    uses_target_pose = _uses_explicit_target_pose(events)
    payload = {
        "ordered_call_skeleton": ordered_call_skeleton,
        "subgoal_signature": _subgoal_signature(
            events,
            has_lift_after_grasp=has_lift,
            uses_explicit_target_pose=uses_target_pose,
        ),
        "has_lift_after_grasp": has_lift,
        "uses_explicit_target_pose": uses_target_pose,
        "orientation_constant": _orientation_constant(tree, events),
        "z_offset_bucket": _z_offset_bucket(_positive_z_offsets(tree)),
        "object_lookup_style": _object_lookup_style(tree, events),
    }
    return {"motif_id": _motif_id(payload), "parse_error": None, **payload}
