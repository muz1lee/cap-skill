from casco.skills.features import analyze_code_features


def test_analyze_code_features_extracts_static_program_traits():
    code = "\n".join(
        [
            "import numpy as np",
            "from math import sqrt",
            "",
            "def helper(x):",
            "    return np.array([sqrt(x)])",
            "",
            "for name in ['bowl']:",
            "    if name:",
            "        goto_pose(helper(1.0), np.eye(4))",
        ]
    )

    assert analyze_code_features(code) == {
        "line_count": 9,
        "nonempty_line_count": 7,
        "function_count": 1,
        "imports": ["math", "numpy"],
        "call_names": ["goto_pose", "helper", "np.array", "np.eye", "sqrt"],
        "uses_conditionals": True,
        "uses_loops": True,
        "has_exception_handler": False,
        "parse_error": None,
    }


def test_analyze_code_features_reports_parse_errors_without_raising():
    features = analyze_code_features("def broken(:\n")

    assert features["parse_error"] is not None
    assert features["function_count"] == 0
    assert features["call_names"] == []
