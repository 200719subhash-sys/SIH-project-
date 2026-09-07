import json

import pytest

from app.catalogue import SchemeDataError, load_catalogue
from app.main import load_schemes


def valid_scheme():
    return {
        "id": "test-scheme",
        "name": "Test Scheme",
        "ministry": "Test Ministry",
        "description": "Test description",
        "benefit": "Test benefit",
        "max_assistance": 1000000,
        "official_url": "https://example.com/scheme",
        "eligibility": {"income": {"max": 500000}},
        "search_text": "test scheme",
        "source": {
            "source_name": "Test fixture",
            "source_type": "other",
            "data_version": "test-1",
            "verification_status": "unverified",
        },
    }


def write_catalogue(tmp_path, schemes=None, **metadata):
    payload = {"scheme_data_version": "test-1", "schemes": schemes or [valid_scheme()]}
    payload.update(metadata)
    path = tmp_path / "schemes.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_current_catalogue_is_versioned_and_unverified():
    catalogue = load_catalogue(__import__("app.main", fromlist=["DATA"]).DATA)
    assert catalogue.scheme_data_version == "phase-2-demo-1"
    assert len(catalogue.schemes) == 6
    assert all(item.source.verification_status == "unverified" for item in catalogue.schemes)
    assert all(item.documents is None for item in catalogue.schemes)
    assert all(item.application_steps is None for item in catalogue.schemes)


@pytest.mark.parametrize(
    ("mutator", "expected"),
    [
        (lambda item: item.pop("id"), "id"),
        (lambda item: item.pop("name"), "name"),
        (lambda item: item.__setitem__("official_url", "not-a-url"), "official_url"),
        (lambda item: item.__setitem__("eligibility", {"income": {"min": 500000, "max": 100000}}), "income.min"),
        (lambda item: item.__setitem__("eligibility", {"min_age": 60, "max_age": 18}), "min_age"),
        (lambda item: item.__setitem__("eligibility", {"income": "invalid"}), "eligibility.income"),
        (lambda item: item.__setitem__("source", {"source_name": "Test", "source_type": "invalid", "data_version": "test-1"}), "source.source_type"),
        (lambda item: item.__setitem__("documents", [{"required": True}]), "documents.0.name"),
    ],
)
def test_invalid_scheme_data_identifies_scheme_and_field(tmp_path, mutator, expected):
    item = valid_scheme()
    mutator(item)
    with pytest.raises(SchemeDataError) as error:
        load_catalogue(write_catalogue(tmp_path, [item]))
    message = str(error.value)
    assert expected in message
    if expected != "id":
        assert "test-scheme" in message
    else:
        assert "scheme[0]" in message


def test_duplicate_scheme_ids_are_rejected(tmp_path):
    first = valid_scheme()
    second = valid_scheme()
    second["name"] = "Second Scheme"
    with pytest.raises(SchemeDataError, match="duplicate scheme IDs: test-scheme"):
        load_catalogue(write_catalogue(tmp_path, [first, second]))


def test_invalid_source_date_range_is_rejected(tmp_path):
    item = valid_scheme()
    item["source"]["effective_from"] = "2026-01-01"
    item["source"]["effective_until"] = "2025-01-01"
    with pytest.raises(SchemeDataError, match="effective_from"):
        load_catalogue(write_catalogue(tmp_path, [item]))


def test_optional_information_remains_unknown(tmp_path):
    scheme = load_catalogue(write_catalogue(tmp_path)).schemes[0]
    assert scheme.documents is None
    assert scheme.application_steps is None
    assert scheme.application_url is None
    assert scheme.implementing_agency is None
    assert scheme.coverage is None
    assert scheme.eligibility.states is None


def test_missing_catalogue_file_is_clear(tmp_path):
    with pytest.raises(SchemeDataError, match="could not load scheme catalogue"):
        load_catalogue(tmp_path / "missing.json")


def test_legacy_loader_returns_scheme_list_for_api_compatibility():
    assert len(load_schemes()) == 6