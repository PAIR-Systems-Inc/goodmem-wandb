"""A failed reranker's vector fallback is reported as vector, and kept.

With ``reranker_id`` set and the reranker failing, GoodMem (v1.0.320) still
returns the vector-stage hits and says so with ``RERANKING_FAILED`` (and
``NOT_FOUND`` naming a missing reranker). 0.2.1 labelled those hits from
configuration: ``score_kind="reranker"`` on vector distances, and
``min_score`` then deleted every hit the server returned.

``retrieve_degraded_hits.ndjson`` and ``retrieve_degraded_empty.ndjson`` are
live captures of that stream (missing reranker
``00000000-0000-7000-8000-000000000000``), shared with camel-goodmem. They are
replayed through the real SDK decoders.
"""

from __future__ import annotations

from typing import Any
import warnings

import pytest

from goodmem_wandb import GoodMemRetrievalModel, GoodMemRetriever, RetrievalHealth
from tests.conftest import Recorder, ndjson_events, ndjson_response

SPACE = "01a0d44b-746f-775b-b91e-bc73d4058e27"
RETRIEVE = "/v1/memories:retrieve"
MISSING_RERANKER = "00000000-0000-7000-8000-000000000000"
WORKING_RERANKER = "019cfda4-7e2f-743c-9edb-e469a97b95c6"
FALLBACK_SCORE = -0.5845972299575806


def serve(recorder: Recorder, events: list[dict[str, Any]]) -> None:
    recorder.route("POST", RETRIEVE, ndjson_response(events))


def retriever(client: Any, **kwargs: Any) -> GoodMemRetriever:
    kwargs.setdefault("reranker_id", MISSING_RERANKER)
    return GoodMemRetriever(space_id=SPACE, client=client, **kwargs)


def codes(out: dict[str, Any]) -> list[str]:
    return [s["code"] for s in out["statuses"]]


def test_the_fixture_is_the_live_degraded_stream():
    events = ndjson_events("retrieve_degraded_hits.ndjson")
    assert [e["status"]["code"] for e in events if "status" in e] == [
        "NOT_FOUND",
        "FEATURE_DISABLED",
        "RERANKING_FAILED",
    ]
    stages = [
        e["resultSetBoundary"]["stageName"] for e in events if "resultSetBoundary" in e
    ]
    assert stages[0] == "retrieve", "the fallback is the vector stage, not 'rerank'"


def test_fallback_hits_are_labelled_vector_not_reranker(recorder, client):
    """0.2.1: score_kind "reranker" on a vector distance of -0.5846."""
    serve(recorder, ndjson_events("retrieve_degraded_hits.ndjson"))
    out = retriever(client).search("canary")

    assert out["score_kind"] == "vector"
    assert [h["score_kind"] for h in out["hits"]] == ["vector"]
    # Raw, as the server sent it: this package never re-orients scores.
    assert out["hits"][0]["score"] == pytest.approx(FALLBACK_SCORE)
    assert out["partial"] is True
    assert codes(out) == ["NOT_FOUND", "RERANKING_FAILED"]


@pytest.mark.parametrize("min_score", [0.0, 0.9, -10.0])
def test_min_score_never_discards_the_fallback_hits(recorder, client, min_score):
    """Contract Q4a: problem + hits -> the hits, partial, statuses. 0.2.1
    applied the reranker threshold to vector distances and returned 0 of 1
    hits with a warning blaming the reranker's scale."""
    serve(recorder, ndjson_events("retrieve_degraded_hits.ndjson"))
    r = retriever(client, min_score=min_score)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = r.search("canary")
    assert not [w for w in caught if "min_score" in str(w.message)]

    assert len(out["hits"]) == 1
    assert "ORYX-2290" in out["hits"][0]["chunk_text"]
    assert out["partial"] is True
    assert codes(out) == ["NOT_FOUND", "RERANKING_FAILED"]


def test_reranking_failed_after_the_hits_still_counts(recorder, client):
    """The decision is made once the whole stream is in."""
    events = ndjson_events("retrieve_vector.ndjson")
    events.append(
        {"status": {"code": "RERANKING_FAILED", "message": "reranker timed out"}}
    )
    serve(recorder, events)
    out = retriever(client, min_score=0.0).search("canary phrase")

    assert out["score_kind"] == "vector"
    assert len(out["hits"]) == 3
    assert all(h["score"] < 0 for h in out["hits"])
    assert out["partial"] is True


@pytest.mark.parametrize(
    "status",
    [
        {
            "code": "NOT_FOUND",
            "message": "Reranker validation failed",
            "details": {"reranker_id": MISSING_RERANKER},
        },
        {
            "code": "NOT_FOUND",
            "message": "lookup failed",
            "details": {"rerankerId": MISSING_RERANKER},
        },
        {"code": "NOT_FOUND", "message": f"Reranker not found: {MISSING_RERANKER}"},
    ],
    ids=["details-reranker_id", "details-rerankerId", "message-only"],
)
def test_a_not_found_naming_the_reranker_alone_means_not_reranked(
    recorder, client, status
):
    events = ndjson_events("retrieve_vector.ndjson")
    events.insert(0, {"status": status})
    serve(recorder, events)
    out = retriever(client, min_score=0.0).search("canary phrase")

    assert out["score_kind"] == "vector"
    assert len(out["hits"]) == 3
    assert codes(out) == ["NOT_FOUND"]


@pytest.mark.parametrize(
    "status",
    [
        {"code": "SOME_FUTURE_CODE", "message": "odd"},
        {
            "code": "NOT_FOUND",
            "message": "Memory not found",
            "details": {"memory_id": "m"},
        },
        {"code": "SUMMARIZATION_FAILED", "message": "llm down"},
    ],
    ids=["unknown-code", "not-found-other", "summarization"],
)
def test_an_unrelated_status_keeps_reranker_scores(recorder, client, status):
    """Control: only a reranker failure turns reranker scores into vector
    ones. Any other problem still marks the result partial."""
    events = ndjson_events("retrieve_reranked.ndjson")
    events.insert(0, {"status": status})
    serve(recorder, events)
    out = retriever(client, reranker_id=WORKING_RERANKER, min_score=0.0).search("q")

    assert out["score_kind"] == "reranker"
    assert [h["score"] for h in out["hits"]] == [pytest.approx(0.21048616, rel=1e-6)]
    assert out["partial"] is True


def test_a_working_reranker_is_unchanged(recorder, client):
    """Control: FEATURE_DISABLED is informational; reranker scores and the
    threshold behave exactly as before."""
    serve(recorder, ndjson_events("retrieve_reranked.ndjson"))
    out = retriever(client, reranker_id=WORKING_RERANKER).search("canary phrase")
    assert out["score_kind"] == "reranker"
    assert all(h["score_kind"] == "reranker" for h in out["hits"])
    assert out["partial"] is False and out["statuses"] == []


def test_a_threshold_that_empties_real_reranker_scores_still_warns(recorder, client):
    """Control: the "removed everything -> name the observed range" warning
    is kept for genuinely reranked results."""
    serve(recorder, ndjson_events("retrieve_reranked.ndjson"))
    with pytest.warns(
        UserWarning, match=r"removed all 3 reranked hit.*-0\.108\.\.0\.210"
    ):
        out = retriever(client, reranker_id=WORKING_RERANKER, min_score=0.9).search("q")
    assert out["hits"] == []


def test_a_failed_reranker_with_no_hits_is_empty_and_flagged(recorder, client):
    """Contract Q4b: problem + no hits -> empty, partial, statuses; no raise
    and no threshold warning."""
    serve(recorder, ndjson_events("retrieve_degraded_empty.ndjson"))
    r = retriever(client, min_score=0.5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = r.search("canary")
    assert not [w for w in caught if "min_score" in str(w.message)]
    assert out["hits"] == []
    assert out["partial"] is True
    assert out["score_kind"] == "vector"
    assert codes(out) == ["NOT_FOUND", "RERANKING_FAILED"]


def test_the_model_and_health_scorer_see_the_fallback(recorder, client):
    serve(recorder, ndjson_events("retrieve_degraded_hits.ndjson"))
    model = GoodMemRetrievalModel(
        space_id=SPACE, client=client, reranker_id=MISSING_RERANKER, min_score=0.5
    )
    out = model.predict("canary")
    assert out["score_kind"] == "vector"
    assert len(out["hits"]) == 1
    assert RetrievalHealth().score(output=out) == {
        "complete": False,
        "num_hits": 1,
        "status_codes": ["NOT_FOUND", "RERANKING_FAILED"],
    }
