"""Retrieval scorers for ``weave.Evaluation``.

Each scorer reads the dict a :class:`~goodmem_wandb.retriever.GoodMemRetriever`
returns and compares it against ground truth carried on the dataset row. None
of them score on the raw relevance number: a GoodMem vector score is an opaque
similarity whose best match may be the lowest value, and a reranker score is a
different scale, so a metric built on the numbers would not mean the same
thing between two configurations. These measure rank and membership, which do.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import weave
from weave.flow.scorer import Scorer


def _hits(output: Any) -> list[dict[str, Any]]:
    if isinstance(output, dict):
        hits = output.get("hits")
        if isinstance(hits, list):
            return hits
    return []


def _ranked_memory_ids(output: Any) -> list[str]:
    """Memory ids in the server's order, first occurrence only.

    Several chunks of one memory are one retrieved document for ranking.
    """
    seen: list[str] = []
    for hit in _hits(output):
        mid = hit.get("memory_id")
        if mid and mid not in seen:
            seen.append(mid)
    return seen


class RecallAtK(Scorer):
    """Fraction of the expected memories that appear in the top ``k``.

    ``k`` counts distinct memories, not chunks.
    """

    k: int = 5

    @weave.op
    def score(
        self, *, output: Any, expected_memory_ids: Sequence[str], **_: Any
    ) -> dict[str, Any]:
        expected = [m for m in (expected_memory_ids or []) if m]
        if not expected:
            # No ground truth on this row: report nothing rather than a
            # perfect or zero score that would move the average.
            return {"recall": None, "found": 0, "expected": 0}
        top = _ranked_memory_ids(output)[: self.k]
        found = sum(1 for m in expected if m in top)
        return {
            "recall": found / len(expected),
            "found": found,
            "expected": len(expected),
        }


class MRR(Scorer):
    """Mean reciprocal rank of the first expected memory.

    ``0.0`` when none of the expected memories was retrieved at all.
    """

    @weave.op
    def score(
        self, *, output: Any, expected_memory_ids: Sequence[str], **_: Any
    ) -> dict[str, Any]:
        expected = {m for m in (expected_memory_ids or []) if m}
        if not expected:
            return {"reciprocal_rank": None, "rank": None}
        for index, memory_id in enumerate(_ranked_memory_ids(output), start=1):
            if memory_id in expected:
                return {"reciprocal_rank": 1.0 / index, "rank": index}
        return {"reciprocal_rank": 0.0, "rank": None}


class FactRecall(Scorer):
    """Did the retrieved text actually contain the fact being looked for?

    An end-to-end check that survives re-chunking and re-embedding, unlike an
    id-based metric. Case-insensitive substring match by default.
    """

    case_sensitive: bool = False

    @weave.op
    def score(self, *, output: Any, expected_text: str, **_: Any) -> dict[str, Any]:
        if not expected_text:
            return {"found": None, "rank": None}
        needle = expected_text if self.case_sensitive else expected_text.lower()
        for index, hit in enumerate(_hits(output), start=1):
            text = hit.get("chunk_text") or ""
            hay = text if self.case_sensitive else text.lower()
            if needle in hay:
                return {"found": True, "rank": index}
        return {"found": False, "rank": None}


class RetrievalHealth(Scorer):
    """Was the retrieval complete, or did the server report a problem?

    Degraded retrievals are easy to miss because they still return results.
    Scoring them makes a run that silently lost a reranker visible as a drop
    in ``complete`` rather than only as a drop in recall.
    """

    @weave.op
    def score(self, *, output: Any, **_: Any) -> dict[str, Any]:
        partial = bool(isinstance(output, dict) and output.get("partial"))
        statuses = output.get("statuses", []) if isinstance(output, dict) else []
        return {
            "complete": not partial,
            "num_hits": len(_hits(output)),
            "status_codes": [s.get("code", "UNKNOWN") for s in statuses],
        }


__all__ = ["MRR", "FactRecall", "RecallAtK", "RetrievalHealth"]
