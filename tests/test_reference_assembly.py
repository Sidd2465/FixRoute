import pytest

from app.schemas import ContextDeeplinkResponse
from app.rule_validation import validate_response_rules
from scripts.demo_reference_assembly import assemble_demo_response, get_row_21_excerpt


def test_row_21_excerpt_contains_full_screen_gesture_section():
    excerpt = get_row_21_excerpt()

    assert "## 4. Full Screen Gesture Function" in excerpt
    assert "Navigation bar" in excerpt
    assert "Select Buttons" in excerpt
    assert "## 5. Touch Sensitivity Setting" not in excerpt
    assert "## 6. Software Updates" not in excerpt
    assert "undamaged charger" not in excerpt.lower()


def test_demo_response_requires_explicit_flag():
    with pytest.raises(ValueError, match="uses-full-screen-gestures"):
        assemble_demo_response(uses_full_screen_gestures=False)


def test_demo_response_builds_real_data_plan_when_flag_is_set():
    demo = assemble_demo_response(uses_full_screen_gestures=True)

    assert demo["selected_deeplink_id"] == "DL-0169"
    assert demo["candidate_ids"][:3] == ["DL-0169", demo["candidate_ids"][1], demo["candidate_ids"][2]]
    assert demo["response"].contexts[0].goal == "Follow these steps to perform this Navigation Configuration"
    assert demo["response"].contexts[0].actions[0].actionName == "Configure Navigation Bar"
    assert len(demo["candidate_scores"]) == len(demo["candidate_ids"])
    assert demo["candidate_scores"][0] > 0

    validated = ContextDeeplinkResponse.model_validate(demo["response"].model_dump(mode="python"))
    assert validate_response_rules(validated) == []
