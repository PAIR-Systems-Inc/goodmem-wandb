"""metadata_filter compares each value as its own type.

0.2.1 turned every value into ``text_equals(field, str(value))``. Live
(v1.0.320), metadata ``{"flag": true, "n": 5}``:

* ``{"flag": True}`` -> ``CAST(val('$.flag') AS TEXT) = 'True'``: HTTP 200,
  0 hits. ``AS BOOLEAN) = true`` matches.
* ``{"n": 5.0}`` -> ``CAST(val('$.n') AS TEXT) = '5.0'``: HTTP 200, 0 hits.
  ``AS NUMERIC) = 5.0`` matches.
* ``{"flag": None}`` -> ``= 'None'``: HTTP 200, 0 hits.

Each looked exactly like "nothing stored". The request bodies below are what
the real SDK put on the wire.
"""

from __future__ import annotations

from typing import Any

import pytest

from goodmem_wandb import GoodMemRetrievalModel, GoodMemRetriever, filters
from tests.conftest import Recorder, ndjson_events, ndjson_response

SPACE = "01a0ce67-ff3e-748d-aba9-0e36a8da091c"
RETRIEVE = "/v1/memories:retrieve"


def sent_filter(recorder: Recorder) -> str:
    return recorder.last_body["spaceKeys"][0]["filter"]


def searched(recorder: Recorder, client: Any, **kwargs: Any) -> str:
    recorder.route(
        "POST", RETRIEVE, ndjson_response(ndjson_events("retrieve_filtered.ndjson"))
    )
    per_call = kwargs.pop("per_call", None)
    GoodMemRetriever(space_id=SPACE, client=client, **kwargs).search(
        "q", metadata_filter=per_call
    )
    return sent_filter(recorder)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, "CAST(val('$.f') AS BOOLEAN) = true"),
        (False, "CAST(val('$.f') AS BOOLEAN) = false"),
        (5, "CAST(val('$.f') AS NUMERIC) = 5"),
        (-3, "CAST(val('$.f') AS NUMERIC) = -3"),
        (5.0, "CAST(val('$.f') AS NUMERIC) = 5.0"),
        (2.5, "CAST(val('$.f') AS NUMERIC) = 2.5"),
        (1e20, "CAST(val('$.f') AS NUMERIC) = 100000000000000000000"),
        (1.5e-7, "CAST(val('$.f') AS NUMERIC) = 0.00000015"),
        ("blue", "CAST(val('$.f') AS TEXT) = 'blue'"),
        ("True", "CAST(val('$.f') AS TEXT) = 'True'"),
        ("5", "CAST(val('$.f') AS TEXT) = '5'"),
    ],
    ids=repr,
)
def test_each_value_is_cast_to_its_own_type(value: Any, expected: str) -> None:
    """bool is checked before int: it is a subclass of int. A string that
    looks like a boolean or a number stays text (control)."""
    assert filters.from_mapping({"f": value}) == expected


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_numbers_are_refused(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        filters.from_mapping({"f": value})


def test_none_is_refused_with_a_clear_error() -> None:
    """0.2.1 sent the text 'None', which matched nothing."""
    with pytest.raises(ValueError, match="'flag' is None"):
        filters.from_mapping({"flag": None})  # type: ignore[dict-item]


@pytest.mark.parametrize("value", [[1], {"a": 1}, b"bytes", (1, 2), object()])
def test_values_without_a_filter_form_are_refused(value: object) -> None:
    with pytest.raises(ValueError, match="Unsupported metadata filter value type"):
        filters.from_mapping({"f": value})  # type: ignore[dict-item]


def test_strings_keep_the_verified_escaping() -> None:
    """Control: backslash escaping, established live, is unchanged."""
    expected = r"CAST(val('$.who') AS TEXT) = 'o\'brien \\ audit'"
    assert filters.from_mapping({"who": "o'brien \\ audit"}) == expected
    assert filters.text_equals("who", "o'brien \\ audit") == expected
    with pytest.raises(ValueError, match="control characters"):
        filters.from_mapping({"who": "a\nb"})
    with pytest.raises(ValueError, match="Unsupported metadata field"):
        filters.from_mapping({"who's": True})


def test_mapping_keeps_insertion_order_and_empty_is_none() -> None:
    assert filters.from_mapping({}) is None
    assert filters.from_mapping({"b": 2, "a": "one"}) == (
        "(CAST(val('$.b') AS NUMERIC) = 2) AND (CAST(val('$.a') AS TEXT) = 'one')"
    )


def test_typed_filters_reach_the_wire(recorder: Recorder, client: Any) -> None:
    sent = searched(
        recorder,
        client,
        metadata_filter={"flag": True, "n": 5, "category": "x"},
    )
    assert sent == (
        "(CAST(val('$.flag') AS BOOLEAN) = true)"
        " AND (CAST(val('$.n') AS NUMERIC) = 5)"
        " AND (CAST(val('$.category') AS TEXT) = 'x')"
    )


def test_a_per_call_filter_is_typed_and_anded(recorder: Recorder, client: Any) -> None:
    sent = searched(
        recorder,
        client,
        metadata_filter={"category": "x"},
        per_call={"archived": False},
        filter="CAST(val('$.year') AS NUMERIC) >= 2026",
    )
    assert sent == (
        "(CAST(val('$.year') AS NUMERIC) >= 2026)"
        " AND (CAST(val('$.category') AS TEXT) = 'x')"
        " AND (CAST(val('$.archived') AS BOOLEAN) = false)"
    )


def test_a_none_value_fails_before_any_request(recorder: Recorder, client: Any) -> None:
    r = GoodMemRetriever(space_id=SPACE, client=client, metadata_filter={"flag": None})
    with pytest.raises(ValueError, match="'flag' is None"):
        r.search("q")
    with pytest.raises(ValueError, match="'flag' is None"):
        GoodMemRetriever(space_id=SPACE, client=client).search(
            "q", metadata_filter={"flag": None}
        )
    assert recorder.requests == []


def test_the_model_sends_typed_filters(recorder: Recorder, client: Any) -> None:
    recorder.route(
        "POST", RETRIEVE, ndjson_response(ndjson_events("retrieve_filtered.ndjson"))
    )
    m = GoodMemRetrievalModel(space_id=SPACE, client=client, metadata_filter={"n": 2.5})
    m.predict("q")
    assert sent_filter(recorder) == "CAST(val('$.n') AS NUMERIC) = 2.5"
