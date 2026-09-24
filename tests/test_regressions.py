"""Every test here fails against goodmem-wandb 0.1.0.

Each one names the behaviour of the live server it pins, and the fixtures are
bytes that server actually sent (v1.0.320).
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from goodmem_wandb import (
    MRR,
    FactRecall,
    GoodMemRetrievalModel,
    GoodMemRetriever,
    GoodMemSpaceError,
    RecallAtK,
    RetrievalHealth,
    filters,
)
from tests.conftest import Recorder, load_json, ndjson_events, ndjson_response

SPACE = "01a0ce67-ff3e-748d-aba9-0e36a8da091c"
RETRIEVE = "/v1/memories:retrieve"


def make(recorder: Recorder, client: Any, fixture: str, **kwargs: Any) -> GoodMemRetriever:
    recorder.route("POST", RETRIEVE, ndjson_response(ndjson_events(fixture)))
    return GoodMemRetriever(space_id=SPACE, client=client, **kwargs)


# --------------------------------------------------------------- scores
def test_vector_scores_are_negative_and_server_order_is_kept(recorder, client):
    """0.1.0 documented relevanceScore as '0-1, higher is better'.

    In this capture the best match scores -0.6154 and is the *lowest* of the
    three. Sorting by score would rank the right answer last.
    """
    r = make(recorder, client, "retrieve_vector.ndjson")
    out = r.search("canary phrase")
    scores = [h["score"] for h in out["hits"]]

    assert scores[0] < 0
    assert scores != sorted(scores, reverse=True)  # not descending
    assert "PLUM-47-HORIZON" in out["hits"][0]["chunk_text"]
    assert out["score_kind"] == "vector"
    assert all(h["score_kind"] == "vector" for h in out["hits"])


def test_reranker_scores_are_a_different_scale_and_also_go_negative(recorder, client):
    """The same three memories, reranked: best is now the *highest* (0.2105),
    and two of the three are still negative -- so '0-1' is wrong here too."""
    r = make(
        recorder, client, "retrieve_reranked.ndjson", reranker_id="019cfda4-7e2f-743c-9edb-e469a97b95c6"
    )
    out = r.search("canary phrase")
    scores = [h["score"] for h in out["hits"]]

    assert out["score_kind"] == "reranker"
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > 0 and min(scores) < 0


def test_min_score_is_ignored_without_a_reranker(recorder, client):
    """A threshold against an opaque vector score is meaningless, so it is not
    applied. 0.1.0 forwarded relevance_threshold regardless, which silently
    emptied the result set."""
    r = make(recorder, client, "retrieve_vector.ndjson", min_score=0.5)
    assert len(r.search("canary phrase")["hits"]) == 3


def test_min_score_applies_with_a_reranker(recorder, client):
    r = make(
        recorder,
        client,
        "retrieve_reranked.ndjson",
        reranker_id="rr",
        min_score=0.0,
    )
    out = r.search("canary phrase")
    assert [h["score"] for h in out["hits"]] == [pytest.approx(0.21048616, rel=1e-6)]


# -------------------------------------------------------------- statuses
def test_feature_disabled_is_not_reported_as_a_failure(recorder, client):
    """Every reranked retrieve without an LLM carries FEATURE_DISABLED. It
    means 'you did not ask for a summary', not 'something went wrong'."""
    raw = ndjson_events("retrieve_reranked.ndjson")
    assert any("status" in e for e in raw), "fixture must contain the status event"

    r = make(recorder, client, "retrieve_reranked.ndjson", reranker_id="rr")
    out = r.search("canary phrase")

    assert out["statuses"] == []
    assert out["partial"] is False
    assert len(out["hits"]) == 3


def test_a_real_status_marks_the_result_partial(recorder, client):
    events = ndjson_events("retrieve_vector.ndjson")
    events.insert(
        0,
        {"status": {"code": "RERANKING_FAILED", "message": "reranker timed out"}},
    )
    recorder.route("POST", RETRIEVE, ndjson_response(events))
    out = GoodMemRetriever(space_id=SPACE, client=client).search("canary phrase")

    assert out["partial"] is True
    assert [s["code"] for s in out["statuses"]] == ["RERANKING_FAILED"]
    assert len(out["hits"]) == 3  # degraded, but the hits are still usable


def test_a_status_the_sdk_does_not_know_is_surfaced_not_swallowed(recorder, client):
    """The SDK decodes an unrecognised code to None. Treating that as harmless
    would let a future server silently break retrieval."""
    events = ndjson_events("retrieve_vector.ndjson")
    events.insert(0, {"status": {"code": "SOME_FUTURE_CODE", "message": "hm"}})
    recorder.route("POST", RETRIEVE, ndjson_response(events))
    out = GoodMemRetriever(space_id=SPACE, client=client).search("q")

    assert out["partial"] is True
    assert out["statuses"][0]["code"] == "UNKNOWN"
    assert out["statuses"][0]["unrecognized"] is True


def test_a_failed_retrieval_is_empty_and_flagged_not_raised(recorder, client):
    """Contract Q4b. 0.1.0 returned success:true, totalResults:0 and blamed
    indexing; the flag and the statuses are what make this distinguishable
    from a genuine miss."""
    recorder.route(
        "POST",
        RETRIEVE,
        ndjson_response([{"status": {"code": "NOT_FOUND", "message": "space gone"}}]),
    )
    out = GoodMemRetriever(space_id=SPACE, client=client).search("q")
    assert out["hits"] == []
    assert out["partial"] is True
    assert [s["code"] for s in out["statuses"]] == ["NOT_FOUND"]


def test_feature_disabled_is_informational_whatever_its_details(recorder, client):
    """Contract Q1: the code alone decides. The server defines FEATURE_DISABLED
    as 'disabled due to missing configuration' -- the caller did not ask for
    the feature -- so no details check is needed or wanted."""
    events = ndjson_events("retrieve_vector.ndjson")
    events.insert(
        0,
        {"status": {"code": "FEATURE_DISABLED", "message": "Reranking disabled: no reranker configured.",
                    "details": {"feature": "reranking", "required_param": "reranker_id"}}},
    )
    recorder.route("POST", RETRIEVE, ndjson_response(events))
    out = GoodMemRetriever(space_id=SPACE, client=client).search("q")
    assert out["statuses"] == []
    assert out["partial"] is False
    assert len(out["hits"]) == 3


def test_a_genuinely_empty_result_is_empty_not_an_error(recorder, client):
    r = make(recorder, client, "retrieve_empty.ndjson")
    out = r.search("xyzzy")
    assert out["hits"] == []
    assert out["partial"] is False
    assert out["statuses"] == []


# --------------------------------------------------------------- secrets
def test_credentials_are_not_weave_fields(client):
    """The finding that shaped this package: weave.publish() uploads a
    weave.Object's pydantic fields verbatim, and weave's should_redact never
    runs on that path. A credential must therefore not be a field at all."""
    r = GoodMemRetriever(
        space_id=SPACE, base_url="https://x", api_key="gm_SECRET_VALUE"
    )
    fields = type(r).model_fields
    assert "api_key" not in fields
    assert "base_url" not in fields
    assert "client" not in fields
    assert "gm_SECRET_VALUE" not in json.dumps(r.model_dump(), default=str)
    assert "gm_SECRET_VALUE" not in repr(r)


def test_publishing_a_retriever_does_not_upload_the_key(weave_recorder, client):
    import weave

    r = GoodMemRetriever(
        space_id=SPACE, base_url="https://x", api_key="gm_SECRET_VALUE", name="r"
    )
    weave.publish(r)
    assert weave_recorder.objs, "publish should have reached the trace server"
    assert "gm_SECRET_VALUE" not in weave_recorder.uploaded_text()
    # …but the configuration that should be versioned did go up.
    assert any(o.get("space_id") == SPACE for o in weave_recorder.objs)


def test_publishing_a_model_does_not_upload_the_key(weave_recorder):
    import weave

    m = GoodMemRetrievalModel(
        space_id=SPACE, base_url="https://x", api_key="gm_SECRET_VALUE", name="m"
    )
    weave.publish(m)
    assert "gm_SECRET_VALUE" not in weave_recorder.uploaded_text()


def test_a_field_literally_named_api_key_would_leak(weave_recorder):
    """Pins why the rule above exists. If this ever starts failing, weave has
    begun redacting published object fields and the constraint can relax."""
    import weave

    class Leaky(weave.Object):
        api_key: str = "gm_SECRET_VALUE"

    weave.publish(Leaky(name="leaky"))
    assert "gm_SECRET_VALUE" in weave_recorder.uploaded_text()


# --------------------------------------------------------------- tracing
def test_search_is_a_weave_op(client):
    from weave.trace.op import is_op

    assert is_op(GoodMemRetriever.search)
    assert is_op(GoodMemRetrievalModel.predict)
    assert is_op(RecallAtK.score)


def test_trace_chunk_text_false_keeps_ids_and_scores_but_drops_text(recorder, client):
    r = make(recorder, client, "retrieve_vector.ndjson", trace_chunk_text=False)
    out = r.search("canary phrase")
    assert all(h["chunk_text"] is None for h in out["hits"])
    assert all(h["chunk_id"] and h["score"] is not None for h in out["hits"])


# ---------------------------------------------------------------- filters
def test_metadata_filter_is_escaped_not_interpolated(recorder, client):
    r = make(
        recorder,
        client,
        "retrieve_filtered.ndjson",
        metadata_filter={"category": "it's \\ bad"},
    )
    r.search("scores")
    sent = recorder.last_body
    expression = sent["spaceKeys"][0]["filter"]
    assert expression == "CAST(val('$.category') AS TEXT) = 'it\\'s \\\\ bad'"


def test_filter_rejects_control_characters():
    """The live grammar rejects a raw newline inside a literal with a 400."""
    with pytest.raises(ValueError, match="control characters"):
        filters.text_equals("category", "a\nb")


def test_filter_rejects_an_unsafe_field_name():
    with pytest.raises(ValueError, match="Unsupported metadata field"):
        filters.text_equals("cat'egory", "x")


def test_no_filter_means_no_filter_key_on_the_space(recorder, client):
    """The SDK normalises space_ids into spaceKeys either way; what must not
    happen is an empty or stray filter riding along."""
    r = make(recorder, client, "retrieve_vector.ndjson")
    r.search("q")
    assert recorder.last_body["spaceKeys"] == [{"spaceId": SPACE}]


# ----------------------------------------------------------------- spaces
def _space_page(items: list[dict[str, Any]]) -> httpx.Response:
    return httpx.Response(200, json={"spaces": items, "nextToken": None})


def _space(name: str, space_id: str, embedder: str) -> dict[str, Any]:
    """A space built from a real captured one, with the identity changed.

    Hand-writing the shape does not work: the SDK validates every field the
    server sends, which is the point of using its decoders in these tests.
    """
    space = json.loads(json.dumps(load_json("spaces_list.json")["spaces"][0]))
    space["spaceId"] = space_id
    space["name"] = name
    for se in space["spaceEmbedders"]:
        se["spaceId"] = space_id
        se["embedderId"] = embedder
    return space


def test_attaching_by_name_reuses_a_space_with_the_same_embedder(recorder, client):
    recorder.route("GET", "/v1/spaces", _space_page([_space("docs", SPACE, "emb-1")]))
    recorder.route("POST", RETRIEVE, ndjson_response(ndjson_events("retrieve_vector.ndjson")))
    r = GoodMemRetriever(space_name="docs", embedder_id="emb-1", client=client)
    assert r.search("q")["space_ids"] == [SPACE]


def test_attaching_by_name_refuses_a_different_embedder(recorder, client):
    """0.1.0 returned reused:true and echoed back the embedder you asked for,
    so every later retrieval silently used the wrong vector space."""
    recorder.route("GET", "/v1/spaces", _space_page([_space("docs", SPACE, "emb-1")]))
    r = GoodMemRetriever(space_name="docs", embedder_id="emb-2", client=client)
    with pytest.raises(GoodMemSpaceError, match="not emb-2"):
        r.search("q")


def test_an_ambiguous_space_name_is_an_error(recorder, client):
    recorder.route(
        "GET",
        "/v1/spaces",
        _space_page([_space("docs", SPACE, "e1"), _space("docs", "other", "e1")]),
    )
    r = GoodMemRetriever(space_name="docs", client=client)
    with pytest.raises(GoodMemSpaceError, match="2 spaces are named"):
        r.search("q")


def test_a_missing_space_is_not_created_unless_asked(recorder, client):
    recorder.route("GET", "/v1/spaces", _space_page([]))
    r = GoodMemRetriever(space_name="nope", client=client)
    with pytest.raises(GoodMemSpaceError, match="create_space=True"):
        r.search("q")


def test_name_filter_matches_are_rechecked_for_an_exact_name(recorder, client):
    """name_filter is a substring match server-side: 'docs' also returns
    'docs-archive', and reusing that would be the wrong space."""
    recorder.route(
        "GET",
        "/v1/spaces",
        _space_page([_space("docs-archive", "other", "e1"), _space("docs", SPACE, "e1")]),
    )
    recorder.route("POST", RETRIEVE, ndjson_response(ndjson_events("retrieve_vector.ndjson")))
    r = GoodMemRetriever(space_name="docs", client=client)
    assert r.search("q")["space_ids"] == [SPACE]


# ------------------------------------------------------------- validation
def test_a_retriever_needs_somewhere_to_search():
    with pytest.raises(ValueError, match="space_id, space_ids or space_name"):
        GoodMemRetriever()


def test_an_empty_query_is_rejected_before_a_request(recorder, client):
    r = GoodMemRetriever(space_id=SPACE, client=client)
    with pytest.raises(ValueError, match="query cannot be empty"):
        r.search("   ")
    assert recorder.requests == []


def test_constructing_a_retriever_makes_no_network_call(recorder, client):
    """0.1.0 did a full GET /v1/spaces in __init__."""
    GoodMemRetriever(space_id=SPACE, client=client)
    assert recorder.requests == []


def test_duplicate_space_ids_are_collapsed_in_order(recorder, client):
    r = GoodMemRetriever(space_id=SPACE, space_ids=[SPACE, "b"], client=client)
    recorder.route("POST", RETRIEVE, ndjson_response(ndjson_events("retrieve_vector.ndjson")))
    assert r.search("q")["space_ids"] == [SPACE, "b"]


def test_fetch_k_widens_the_candidate_set_before_reranking(recorder, client):
    r = make(recorder, client, "retrieve_reranked.ndjson", reranker_id="rr", limit=2, fetch_k=20)
    out = r.search("q")
    config = recorder.last_body["postProcessor"]["config"]
    assert recorder.last_body["requestedSize"] == 20
    assert config == {"reranker_id": "rr", "max_results": 2}
    # No relevance_threshold is ever sent: 0.1.0 forwarded one and the server
    # silently dropped every result. Thresholding happens here, visibly.
    assert "relevance_threshold" not in config
    assert len(out["hits"]) == 2


# ----------------------------------------------------------------- scorers
def _result(memory_ids: list[str], **extra: Any) -> dict[str, Any]:
    return {
        "hits": [
            {"memory_id": m, "chunk_id": f"c{i}", "chunk_text": f"text {m}", "score": 0.1}
            for i, m in enumerate(memory_ids)
        ],
        **extra,
    }


def test_recall_at_k_counts_distinct_memories_not_chunks():
    out = {
        "hits": [
            {"memory_id": "m1", "chunk_id": "c1", "chunk_text": "a"},
            {"memory_id": "m1", "chunk_id": "c2", "chunk_text": "b"},
            {"memory_id": "m2", "chunk_id": "c3", "chunk_text": "c"},
        ]
    }
    assert RecallAtK(k=2).score(output=out, expected_memory_ids=["m1", "m2"]) == {
        "recall": 1.0,
        "found": 2,
        "expected": 2,
    }


def test_recall_reports_none_without_ground_truth():
    """A row with no expected ids must not count as a perfect or a zero score."""
    assert RecallAtK().score(output=_result(["m1"]), expected_memory_ids=[])["recall"] is None


def test_mrr_uses_the_rank_of_the_first_expected_memory():
    assert MRR().score(output=_result(["m1", "m2", "m3"]), expected_memory_ids=["m3"]) == {
        "reciprocal_rank": pytest.approx(1 / 3),
        "rank": 3,
    }
    assert MRR().score(output=_result(["m1"]), expected_memory_ids=["zz"]) == {
        "reciprocal_rank": 0.0,
        "rank": None,
    }


def test_fact_recall_finds_the_fact_in_the_retrieved_text(recorder, client):
    r = make(recorder, client, "retrieve_vector.ndjson")
    out = r.search("canary phrase")
    assert FactRecall().score(output=out, expected_text="plum-47-horizon") == {
        "found": True,
        "rank": 1,
    }


def test_retrieval_health_reports_degradation_as_a_metric():
    degraded = _result(["m1"], partial=True, statuses=[{"code": "RERANKING_FAILED"}])
    assert RetrievalHealth().score(output=degraded) == {
        "complete": False,
        "num_hits": 1,
        "status_codes": ["RERANKING_FAILED"],
    }


# ------------------------------------------------------------------ model
def test_the_model_predicts_through_the_retriever(recorder, client):
    recorder.route("POST", RETRIEVE, ndjson_response(ndjson_events("retrieve_vector.ndjson")))
    m = GoodMemRetrievalModel(space_id=SPACE, client=client)
    out = m.predict("canary phrase")
    assert len(out["hits"]) == 3
    assert out["score_kind"] == "vector"


def test_the_model_versions_its_retrieval_configuration():
    m = GoodMemRetrievalModel(space_id=SPACE, reranker_id="rr", limit=7, api_key="gm_X")
    dumped = m.model_dump()
    assert dumped["reranker_id"] == "rr"
    assert dumped["limit"] == 7
    assert "gm_X" not in json.dumps(dumped, default=str)


# ------------------------------------------------------------- fixture use
def test_fixtures_are_real_server_bytes():
    """Guards against a fixture being hand-written later."""
    space = load_json("space.json")
    assert space["spaceId"] == SPACE
    events = ndjson_events("retrieve_reranked.ndjson")
    codes = [e["status"]["code"] for e in events if "status" in e]
    assert codes == ["FEATURE_DISABLED"]
    assert any(e.get("resultSetBoundary", {}).get("stageName") == "rerank" for e in events)


def test_a_min_score_that_removes_every_reranked_hit_says_so(recorder, client):
    """Measured live 2026-09-24: Voyage rerank-2.5 0.27..0.93, Jina
    jina-reranker-v3 -0.14..0.43 on the same documents. A threshold tuned
    for one empties the other; the empty case must not look like a miss."""
    r = make(recorder, client, "retrieve_reranked.ndjson", reranker_id="rr", min_score=0.9)
    with pytest.warns(UserWarning, match="removed all 3 reranked hit"):
        out = r.search("canary phrase")
    assert out["hits"] == []
    assert out["partial"] is False, "the threshold, not the server, emptied it"
