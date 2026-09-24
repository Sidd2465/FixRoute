import json

import pytest

from app.data_loader import (
    load_complaints,
    load_deeplinks,
    load_references,
)


def test_load_complaints_success_and_blank_line_handling(tmp_path):
    complaint_file = tmp_path / "input.txt"
    complaint_file.write_text(
        "First complaint\n\nSecond complaint with extra spaces   \n\n\nThird complaint\n",
        encoding="utf-8",
    )

    complaints = load_complaints(complaint_file)

    assert complaints == [
        "First complaint",
        "Second complaint with extra spaces",
        "Third complaint",
    ]


def test_load_complaints_preserves_numbered_symptom_lines(tmp_path):
    complaint_file = tmp_path / "input.txt"
    complaint_file.write_text(
        "1. My screen is black.\n2. The touch stops working.\n",
        encoding="utf-8",
    )

    complaints = load_complaints(complaint_file)

    assert complaints == ["1. My screen is black.", "2. The touch stops working."]


def test_load_complaints_keeps_multiple_numbered_symptoms_on_one_line(tmp_path):
    complaint_file = tmp_path / "input.txt"
    complaint_file.write_text(
        "1. Screen is black. 2. Touch is unresponsive. 3. Device overheats.\n",
        encoding="utf-8",
    )

    complaints = load_complaints(complaint_file)

    assert complaints == ["1. Screen is black. 2. Touch is unresponsive. 3. Device overheats."]


def test_load_references_success(tmp_path):
    catalog = tmp_path / "siis_responses.json"
    catalog.write_text(
        json.dumps(
            {
                "count": 1,
                "responses": [
                    {
                        "id": "row_1",
                        "original_query": "Screen flickers",
                        "siis_response": {
                            "title": "Display issue",
                            "content": "Try restarting the device.",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    references = load_references(catalog)

    assert references == [
        {
            "id": "row_1",
            "original_query": "Screen flickers",
            "siis_response": {
                "title": "Display issue",
                "content": "Try restarting the device.",
            },
        }
    ]


def test_load_deeplinks_success(tmp_path):
    fixture = {
        "id": "DL-0001",
        "deeplink": "bixby://masked/act/example",
        "description": "Opens settings",
        "message": "Open settings",
        "originalType": "onURL",
        "control_type": None,
        "qna_description": "An accessible setting",
        "validation": {
            "deeplink": "bixby://masked/val/example",
            "key": "Use setting",
            "resultType": "boolean",
            "condition": "equal",
            "value": "True",
        },
        "extra": {"enabled": True, "priority": 2},
    }
    catalog = tmp_path / "deeplinks.json"
    catalog.write_text(
        json.dumps({"count": 1, "deeplinks": [fixture]}),
        encoding="utf-8",
    )

    deeplinks = load_deeplinks(catalog)

    assert deeplinks == [fixture]


def test_load_references_invalid_json(tmp_path):
    catalog = tmp_path / "siis_responses.json"
    catalog.write_text('{"responses": [}', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        load_references(catalog)


def test_load_references_missing_fields(tmp_path):
    catalog = tmp_path / "siis_responses.json"
    catalog.write_text(
        json.dumps(
            {
                "count": 1,
                "responses": [
                    {
                        "id": "row_1",
                        "original_query": "Screen flickers",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="siis_response"):
        load_references(catalog)


@pytest.mark.parametrize(
    ("bad_count", "expected_message"),
    [
        pytest.param("missing", "must include an integer 'count' field", id="missing"),
        pytest.param(None, "expected int", id="null"),
        pytest.param(True, "expected int", id="bool"),
        pytest.param(1.5, "expected int", id="float"),
        pytest.param("2", "expected int", id="string"),
        pytest.param(-1, "negative 'count' value", id="negative"),
    ],
)
def test_load_references_invalid_count_values(tmp_path, bad_count, expected_message):
    catalog = tmp_path / "siis_responses.json"
    payload = {
        "responses": [
            {
                "id": "row_1",
                "original_query": "Screen flickers",
                "siis_response": {
                    "title": "Display issue",
                    "content": "Try restarting the device.",
                },
            }
        ]
    }
    if bad_count == "missing":
        payload.pop("count", None)
    else:
        payload["count"] = bad_count

    catalog.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=expected_message):
        load_references(catalog)


def test_load_references_count_mismatch(tmp_path):
    catalog = tmp_path / "siis_responses.json"
    catalog.write_text(
        json.dumps(
            {
                "count": 2,
                "responses": [
                    {
                        "id": "row_1",
                        "original_query": "Screen flickers",
                        "siis_response": {
                            "title": "Display issue",
                            "content": "Try restarting the device.",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Count mismatch"):
        load_references(catalog)


def test_load_references_duplicate_ids(tmp_path):
    catalog = tmp_path / "siis_responses.json"
    catalog.write_text(
        json.dumps(
            {
                "count": 2,
                "responses": [
                    {
                        "id": "row_1",
                        "original_query": "Screen flickers",
                        "siis_response": {
                            "title": "Display issue",
                            "content": "Try restarting the device.",
                        },
                    },
                    {
                        "id": "row_1",
                        "original_query": "Screen turn-off",
                        "siis_response": {
                            "title": "Another issue",
                            "content": "Try factory reset.",
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate reference id"):
        load_references(catalog)


def test_load_references_invalid_utf8(tmp_path):
    catalog = tmp_path / "siis_responses.json"
    catalog.write_bytes(b"\xff\xfe\x00\x00")

    with pytest.raises(ValueError, match="invalid UTF-8"):
        load_references(catalog)


def test_load_deeplinks_missing_fields(tmp_path):
    catalog = tmp_path / "deeplinks.json"
    catalog.write_text(
        json.dumps(
            {
                "count": 1,
                "deeplinks": [
                    {
                        "id": "DL-0001",
                        "description": "Opens settings",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="deeplink"):
        load_deeplinks(catalog)


@pytest.mark.parametrize(
    ("bad_count", "expected_message"),
    [
        pytest.param("missing", "must include an integer 'count' field", id="missing"),
        pytest.param(None, "expected int", id="null"),
        pytest.param(True, "expected int", id="bool"),
        pytest.param(1.5, "expected int", id="float"),
        pytest.param("2", "expected int", id="string"),
        pytest.param(-1, "negative 'count' value", id="negative"),
    ],
)
def test_load_deeplinks_invalid_count_values(tmp_path, bad_count, expected_message):
    catalog = tmp_path / "deeplinks.json"
    payload = {
        "deeplinks": [
            {
                "id": "DL-0001",
                "deeplink": "bixby://masked/act/example",
                "description": "Opens settings",
            }
        ]
    }
    if bad_count == "missing":
        payload.pop("count", None)
    else:
        payload["count"] = bad_count

    catalog.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=expected_message):
        load_deeplinks(catalog)


def test_load_deeplinks_count_mismatch(tmp_path):
    catalog = tmp_path / "deeplinks.json"
    catalog.write_text(
        json.dumps(
            {
                "count": 2,
                "deeplinks": [
                    {
                        "id": "DL-0001",
                        "deeplink": "bixby://masked/act/example",
                        "description": "Opens settings",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Count mismatch"):
        load_deeplinks(catalog)


def test_load_deeplinks_duplicate_ids(tmp_path):
    catalog = tmp_path / "deeplinks.json"
    catalog.write_text(
        json.dumps(
            {
                "count": 2,
                "deeplinks": [
                    {
                        "id": "DL-0001",
                        "deeplink": "bixby://masked/act/example",
                        "description": "Opens settings",
                    },
                    {
                        "id": "DL-0001",
                        "deeplink": "bixby://masked/act/other",
                        "description": "Other settings",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate deeplink id"):
        load_deeplinks(catalog)


def test_missing_file_raises_clear_error(tmp_path):
    missing = tmp_path / "does_not_exist.json"

    with pytest.raises(FileNotFoundError):
        load_references(missing)
