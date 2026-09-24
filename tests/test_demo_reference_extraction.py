from scripts import demo_reference_extraction as demo


def test_no_live_guard_exits_without_google_access(capsys, monkeypatch):
    def fail_load_api_key():
        raise AssertionError("dotenv should not be loaded without --run-live")

    def fail_google_client():
        raise AssertionError("Google client should not be opened without --run-live")

    monkeypatch.setattr(demo, "_load_api_key", fail_load_api_key)
    monkeypatch.setattr(demo, "_open_google_client", fail_google_client)

    exit_code = demo.main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "--run-live" in captured.out
    assert "without --run-live" in captured.out


def test_row_21_full_screen_section_is_selected():
    section = demo.get_full_screen_gesture_section(reference_id="row_21")

    assert section.startswith("## 4. Full Screen Gesture Function")
    assert "## 5. Touch Sensitivity Setting" not in section
    assert "## 3." not in section
    assert "## 4. Full Screen Gesture Function" in section
