import json
from pathlib import Path

import pytest

from app.data_loader import load_deeplinks
from app.retrieval import DeeplinkRetriever


@pytest.fixture
def synthetic_catalogue():
    return [
        {
            "id": "DL-1001",
            "deeplink": "bixby://masked/act/enable_wifi",
            "description": "Opens the Wi-Fi settings page on the device.",
            "message": "Open Wi-Fi settings",
            "qna_description": "Show Wi-Fi connection settings and available networks.",
            "originalType": "onClickURL",
            "extra": {"group": "network"},
        },
        {
            "id": "DL-1002",
            "deeplink": "bixby://masked/act/open_nav_bar",
            "description": "Opens the navigation bar settings panel.",
            "message": "Open navigation bar settings",
            "qna_description": "Configure the navigation bar, buttons, and gestures.",
            "originalType": "onClickURL",
            "extra": {"group": "accessibility"},
        },
        {
            "id": "DL-1003",
            "deeplink": "bixby://masked/act/enable_backup",
            "description": "Enables Samsung Cloud backup in device settings.",
            "message": "Enable Samsung Cloud backup",
            "qna_description": "Turn on backup to Samsung Cloud for device data protection.",
            "originalType": "onURL",
            "extra": {"group": "backup"},
        },
        {
            "id": "DL-1004",
            "deeplink": "bixby://masked/act/disable_backup",
            "description": "Disables Samsung Cloud backup in device settings.",
            "message": "Disable Samsung Cloud backup",
            "qna_description": "Turn off Samsung Cloud backup for device data protection.",
            "originalType": "offURL",
            "extra": {"group": "backup"},
        },
        {
            "id": "DL-1005",
            "deeplink": "bixby://masked/act/other_feature",
            "description": "Shows battery optimization settings.",
            "message": "Open battery settings",
            "qna_description": "Manage battery saver and optimization features.",
            "originalType": "onClickURL",
            "extra": {"group": "power"},
        },
    ]


def test_search_ranks_relevant_entries(synthetic_catalogue):
    retriever = DeeplinkRetriever(synthetic_catalogue)

    results = retriever.search("Open Wi-Fi settings")

    assert results[0]["entry"]["id"] == "DL-1001"
    assert results[0]["similarity_score"] > 0


def test_search_is_case_insensitive_and_preserves_metadata(synthetic_catalogue):
    retriever = DeeplinkRetriever(synthetic_catalogue)

    results = retriever.search("ENABLE SAMSUNG CLOUD BACKUP")

    assert results[0]["entry"]["id"] == "DL-1003"
    assert results[0]["entry"]["deeplink"] == "bixby://masked/act/enable_backup"
    assert results[0]["entry"]["extra"] == {"group": "backup"}


def test_search_handles_accents_consistently():
    catalogue = [
        {
            "id": "DL-2001",
            "deeplink": "bixby://masked/act/cafe_settings",
            "description": "Opens the café settings page.",
            "message": "Open café settings",
            "qna_description": "Configure café preferences and display options.",
            "originalType": "onClickURL",
            "extra": {"group": "settings"},
        },
        {
            "id": "DL-2002",
            "deeplink": "bixby://masked/act/battery_settings",
            "description": "Shows battery optimization settings.",
            "message": "Open battery settings",
            "qna_description": "Manage battery saver and optimization features.",
            "originalType": "onClickURL",
            "extra": {"group": "power"},
        },
    ]
    retriever = DeeplinkRetriever(catalogue)

    accented = retriever.search("Café settings")
    plain = retriever.search("Cafe settings")

    assert accented[0]["entry"]["id"] == "DL-2001"
    assert plain[0]["entry"]["id"] == "DL-2001"
    assert accented[0]["similarity_score"] == pytest.approx(plain[0]["similarity_score"])


def test_search_top_k_limit(synthetic_catalogue):
    retriever = DeeplinkRetriever(synthetic_catalogue)

    results = retriever.search("settings", top_k=2)

    assert len(results) <= 2


def test_blank_query_returns_empty_list(synthetic_catalogue):
    retriever = DeeplinkRetriever(synthetic_catalogue)

    assert retriever.search("   ") == []


def test_no_overlap_query_returns_empty_list(synthetic_catalogue):
    retriever = DeeplinkRetriever(synthetic_catalogue)

    assert retriever.search("quantum gravity reactor") == []


def test_invalid_top_k_rejected(synthetic_catalogue):
    retriever = DeeplinkRetriever(synthetic_catalogue)

    for bad_value in (True, 0, -1, 1.5, "3"):
        with pytest.raises(ValueError, match="positive integer"):
            retriever.search("open settings", top_k=bad_value)


def test_empty_catalogue_rejected():
    with pytest.raises(ValueError, match="empty"):
        DeeplinkRetriever([])


def test_deterministic_ordering_by_id_on_ties():
    catalogue = [
        {
            "id": "DL-1002",
            "deeplink": "bixby://masked/act/tied_action",
            "description": "Open settings",
            "message": "Open settings",
            "qna_description": "Open settings",
            "originalType": "onClickURL",
            "extra": {"group": "settings"},
        },
        {
            "id": "DL-1001",
            "deeplink": "bixby://masked/act/tied_action_two",
            "description": "Open settings",
            "message": "Open settings",
            "qna_description": "Open settings",
            "originalType": "onClickURL",
            "extra": {"group": "settings"},
        },
    ]
    retriever = DeeplinkRetriever(catalogue)

    first = retriever.search("settings")
    second = retriever.search("settings")

    assert [item["entry"]["id"] for item in first] == ["DL-1001", "DL-1002"]
    assert [item["entry"]["id"] for item in first] == [item["entry"]["id"] for item in second]


def test_input_entries_are_not_mutated(synthetic_catalogue):
    original = json.dumps(synthetic_catalogue, sort_keys=True)
    retriever = DeeplinkRetriever(synthetic_catalogue)

    retriever.search("open navigation settings")

    assert json.dumps(synthetic_catalogue, sort_keys=True) == original


def test_catalogue_changes_after_construction_are_ignored(synthetic_catalogue):
    retriever = DeeplinkRetriever(synthetic_catalogue)

    synthetic_catalogue[0]["description"] = "Changed description"
    synthetic_catalogue[0]["extra"]["group"] = "changed"

    assert retriever.entries[0]["description"] == "Opens the Wi-Fi settings page on the device."
    assert retriever.entries[0]["extra"] == {"group": "network"}
    assert retriever.search("Open Wi-Fi settings")[0]["entry"]["id"] == "DL-1001"


def test_result_metadata_is_not_shared_across_searches(synthetic_catalogue):
    retriever = DeeplinkRetriever(synthetic_catalogue)

    first = retriever.search("Open Wi-Fi settings")
    first[0]["entry"]["extra"]["group"] = "mutated"

    second = retriever.search("Open Wi-Fi settings")
    assert first[0]["entry"]["extra"] == {"group": "mutated"}
    assert second[0]["entry"]["extra"] == {"group": "network"}


def test_real_catalogue_examples_are_ranked_and_reported():
    catalog_path = Path(__file__).resolve().parent.parent / "student_kit" / "deeplinks.json"
    entries = load_deeplinks(catalog_path)
    retriever = DeeplinkRetriever(entries)

    checks = {
        "Open Navigation bar settings": "DL-0169",
        "Open Wi-Fi settings": "DL-0313",
        "Open 24-hour time format settings": "DL-0001",
        "Enable Samsung Cloud backup": "DL-0542",
        "Disable Samsung Cloud backup": "DL-0541",
    }

    for query, expected_id in checks.items():
        ranked = retriever.search(query)
        assert ranked, f"No results for query: {query}"
        ids = [item["entry"]["id"] for item in ranked]
        assert expected_id in ids, f"Expected {expected_id} for {query!r}, got {ids[:5]}"
