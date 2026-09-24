"""Live end-to-end tests. Skipped unless a server is configured.

    GOODMEM_BASE_URL=https://localhost:8080 \
    GOODMEM_API_KEY=gm_… \
    GOODMEM_EMBEDDER_ID=… \
    GOODMEM_VERIFY_SSL=0 \
    pytest -m integration

There is no default credential. Everything created here is deleted in the
fixture teardown and the teardown asserts it is gone.
"""

from __future__ import annotations

import os
import time
from typing import Any
import uuid

from goodmem import Goodmem
import pytest

from goodmem_wandb import (
    MRR,
    FactRecall,
    GoodMemRetrievalModel,
    GoodMemRetriever,
    GoodMemSpaceError,
    RecallAtK,
    RetrievalHealth,
)
from goodmem_wandb._spaces import find_by_name

pytestmark = pytest.mark.integration

BASE_URL = os.getenv("GOODMEM_BASE_URL")
API_KEY = os.getenv("GOODMEM_API_KEY")
EMBEDDER_ID = os.getenv("GOODMEM_EMBEDDER_ID")
RERANKER_ID = os.getenv("GOODMEM_RERANKER_ID")
VERIFY_SSL = os.getenv("GOODMEM_VERIFY_SSL", "1") not in ("0", "false", "False")

if not (BASE_URL and API_KEY and EMBEDDER_ID):
    pytest.skip(
        "set GOODMEM_BASE_URL, GOODMEM_API_KEY and GOODMEM_EMBEDDER_ID",
        allow_module_level=True,
    )

CANARY = "The GoodMem Weave live-test canary is FIG-88-MERIDIAN."
FACTS = [
    (CANARY, {"category": "canary"}),
    ("Weave records an op call's inputs, output and latency.", {"category": "weave"}),
    ("A GoodMem vector score may be negative.", {"category": "scores"}),
]


@pytest.fixture(scope="module")
def live() -> Any:
    client = Goodmem(base_url=BASE_URL, api_key=API_KEY, verify=VERIFY_SSL)
    name = f"goodmem-wandb-e2e-{uuid.uuid4().hex[:8]}"
    space = client.spaces.create(
        name=name,
        space_embedders=[{"embedderId": EMBEDDER_ID, "defaultRetrievalWeight": 1.0}],
        default_chunking_config={
            "recursive": {
                "chunkSize": 256,
                "chunkOverlap": 25,
                "separators": ["\n\n", "\n", ". ", " ", ""],
                "keepStrategy": "KEEP_END",
                "separatorIsRegex": False,
                "lengthMeasurement": "CHARACTER_COUNT",
            }
        },
    )
    memory_ids = []
    for text, metadata in FACTS:
        memory = client.memories.create(
            space_id=space.space_id,
            content_type="text/plain",
            original_content=text,
            metadata=metadata,
        )
        memory_ids.append(memory.memory_id)

    deadline = time.time() + 180
    while time.time() < deadline:
        states = [
            client.memories.get(id=m).processing_status for m in memory_ids
        ]
        if all(s == "COMPLETED" for s in states):
            break
        if any(s == "FAILED" for s in states):
            pytest.fail(f"ingestion failed: {states}")
        time.sleep(2)
    else:
        pytest.fail("memories did not finish indexing within 180s")

    yield {
        "client": client,
        "space_id": space.space_id,
        "space_name": name,
        "memory_ids": memory_ids,
        "canary_id": memory_ids[0],
    }

    client.spaces.delete(id=space.space_id)
    assert find_by_name(client, name) == [], "teardown left the space behind"
    client.close()


@pytest.fixture
def retriever(live: Any) -> GoodMemRetriever:
    return GoodMemRetriever(space_id=live["space_id"], client=live["client"], limit=5)


def test_exact_fact_is_recalled(retriever: GoodMemRetriever) -> None:
    out = retriever.search("what is the live-test canary")
    assert any("FIG-88-MERIDIAN" in h["chunk_text"] for h in out["hits"])
    assert out["score_kind"] == "vector"
    assert out["partial"] is False


def test_vector_scores_come_back_as_the_server_sent_them(
    retriever: GoodMemRetriever,
) -> None:
    out = retriever.search("canary")
    scores = [h["score"] for h in out["hits"]]
    assert scores, "expected hits"
    # Documented as 0-1 in 0.1.0; on a live server they are not.
    assert any(s < 0 for s in scores)


def test_feature_disabled_does_not_look_like_a_failure(live: Any) -> None:
    if not RERANKER_ID:
        pytest.skip("set GOODMEM_RERANKER_ID")
    r = GoodMemRetriever(
        space_id=live["space_id"], client=live["client"], reranker_id=RERANKER_ID
    )
    out = r.search("canary")
    assert out["hits"]
    assert out["statuses"] == []
    assert out["partial"] is False
    assert out["score_kind"] == "reranker"


def test_metadata_filter_scopes_the_search(retriever: GoodMemRetriever) -> None:
    out = retriever.search("anything", metadata_filter={"category": "scores"})
    assert out["hits"]
    assert all(h["metadata"].get("category") == "scores" for h in out["hits"])


def test_a_filter_matching_nothing_is_empty_not_an_error(
    retriever: GoodMemRetriever,
) -> None:
    out = retriever.search("canary", metadata_filter={"category": "no-such-category"})
    assert out["hits"] == []
    assert out["partial"] is False


def test_a_quote_in_a_filter_value_is_escaped_not_rejected(
    retriever: GoodMemRetriever,
) -> None:
    """A 400 here would mean the escaping is wrong for the live grammar."""
    out = retriever.search("canary", metadata_filter={"category": "it's \\ odd"})
    assert out["hits"] == []


def test_attaching_by_name_reuses_the_space(live: Any) -> None:
    r = GoodMemRetriever(
        space_name=live["space_name"],
        embedder_id=EMBEDDER_ID,
        client=live["client"],
    )
    assert r.search("canary")["space_ids"] == [live["space_id"]]


def test_attaching_by_name_refuses_a_different_embedder(live: Any) -> None:
    r = GoodMemRetriever(
        space_name=live["space_name"],
        embedder_id="00000000-0000-0000-0000-000000000000",
        client=live["client"],
    )
    with pytest.raises(GoodMemSpaceError, match="Retrieval across mismatched"):
        r.search("canary")


def test_the_model_and_scorers_run_over_a_real_retrieval(live: Any) -> None:
    model = GoodMemRetrievalModel(space_id=live["space_id"], client=live["client"])
    out = model.predict("what is the live-test canary")

    assert RecallAtK(k=5).score(
        output=out, expected_memory_ids=[live["canary_id"]]
    )["recall"] == 1.0
    assert MRR().score(output=out, expected_memory_ids=[live["canary_id"]])[
        "reciprocal_rank"
    ] > 0
    assert FactRecall().score(output=out, expected_text="FIG-88-MERIDIAN")["found"]
    assert RetrievalHealth().score(output=out)["complete"] is True


def test_a_missing_space_raises_rather_than_returning_nothing(live: Any) -> None:
    r = GoodMemRetriever(
        space_id="00000000-0000-0000-0000-000000000000", client=live["client"]
    )
    with pytest.raises(Exception, match=r"(?i)not.?found|permission|denied|invalid"):
        r.search("canary")


def test_trace_chunk_text_false_still_retrieves(live: Any) -> None:
    r = GoodMemRetriever(
        space_id=live["space_id"], client=live["client"], trace_chunk_text=False
    )
    out = r.search("canary")
    assert out["hits"]
    assert all(h["chunk_text"] is None for h in out["hits"])
    assert all(h["score"] is not None for h in out["hits"])
