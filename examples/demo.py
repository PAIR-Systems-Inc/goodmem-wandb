"""Trace a GoodMem retrieval in Weave, then evaluate two configurations.

    GOODMEM_BASE_URL=https://localhost:8080 \
    GOODMEM_API_KEY=gm_… \
    GOODMEM_SPACE_ID=… \
    python examples/demo.py
"""

from __future__ import annotations

import os

import weave

from goodmem_wandb import (
    MRR,
    FactRecall,
    GoodMemRetrievalModel,
    GoodMemRetriever,
    RecallAtK,
    RetrievalHealth,
)

SPACE_ID = os.environ["GOODMEM_SPACE_ID"]
RERANKER_ID = os.getenv("GOODMEM_RERANKER_ID")

weave.init(os.getenv("WEAVE_PROJECT", "goodmem-demo"))

# 1. One traced retrieval. Open the printed Weave URL to see the call.
retriever = GoodMemRetriever(space_id=SPACE_ID, limit=5)
result = retriever.search("what is this corpus about?")

print(f"score_kind={result['score_kind']}  partial={result['partial']}")
for hit in result["hits"]:
    print(f"  {hit['score']:+.4f}  {hit['chunk_text'][:70]}")
if result["statuses"]:
    print("  server statuses:", [s["code"] for s in result["statuses"]])

# 2. Compare two retrieval configurations over a small dataset.
dataset = [
    {
        "question": "what is this corpus about?",
        "expected_memory_ids": [hit["memory_id"] for hit in result["hits"][:1]],
        "expected_text": (result["hits"][0]["chunk_text"][:30] if result["hits"] else ""),
    }
]
evaluation = weave.Evaluation(
    dataset=dataset,
    scorers=[RecallAtK(k=5), MRR(), FactRecall(), RetrievalHealth()],
)

evaluation.evaluate(GoodMemRetrievalModel(space_id=SPACE_ID, limit=5))
if RERANKER_ID:
    evaluation.evaluate(
        GoodMemRetrievalModel(space_id=SPACE_ID, limit=5, reranker_id=RERANKER_ID)
    )
