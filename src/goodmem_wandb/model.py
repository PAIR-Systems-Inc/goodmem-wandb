"""A weave.Model wrapper so a GoodMem retriever can be evaluated."""

from __future__ import annotations

from typing import Any

from pydantic import PrivateAttr
import weave

from goodmem_wandb._connection import GoodMemConnection, split_connection_kwargs
from goodmem_wandb.retriever import GoodMemRetriever


class GoodMemRetrievalModel(weave.Model):
    """Wrap a :class:`GoodMemRetriever` as something ``weave.Evaluation`` can run.

    ``predict`` takes one question and returns the retriever's result, so the
    scorers in :mod:`goodmem_wandb.scorers` can read ``hits`` from it::

        import weave
        from goodmem_wandb import GoodMemRetrievalModel, RecallAtK, MRR

        weave.init("my-project")
        model = GoodMemRetrievalModel(space_name="docs", limit=5)

        weave.Evaluation(
            dataset=[{"question": "how do I rotate a key?",
                      "expected_memory_ids": ["01a0…"]}],
            scorers=[RecallAtK(k=5), MRR()],
        ).evaluate(model)

    Changing any field here versions the model in Weave, so two retrieval
    configurations can be compared directly. Credentials are not fields and
    are therefore not part of the version -- see
    :class:`goodmem_wandb._connection.GoodMemConnection`.
    """

    space_id: str | None = None
    space_ids: list[str] | None = None
    space_name: str | None = None
    embedder_id: str | None = None
    limit: int = 5
    fetch_k: int | None = None
    reranker_id: str | None = None
    min_score: float | None = None
    filter: str | None = None
    metadata_filter: dict[str, Any] | None = None
    llm_id: str | None = None
    llm_temperature: float | None = None
    trace_chunk_text: bool = True

    _conn: GoodMemConnection = PrivateAttr()
    _retriever: GoodMemRetriever = PrivateAttr()

    def __init__(self, **data: Any) -> None:
        conn = split_connection_kwargs(data)
        super().__init__(**data)
        self._conn = conn
        self._retriever = GoodMemRetriever(
            space_id=self.space_id,
            space_ids=list(self.space_ids or []),
            space_name=self.space_name,
            embedder_id=self.embedder_id,
            limit=self.limit,
            fetch_k=self.fetch_k,
            reranker_id=self.reranker_id,
            min_score=self.min_score,
            filter=self.filter,
            metadata_filter=self.metadata_filter,
            llm_id=self.llm_id,
            llm_temperature=self.llm_temperature,
            trace_chunk_text=self.trace_chunk_text,
            **conn.model_dump(exclude_none=True, exclude={"api_key", "client"}),
            api_key=conn.api_key,
            client=conn.client,
        )

    @property
    def retriever(self) -> GoodMemRetriever:
        """The underlying retriever, for use outside an evaluation."""
        return self._retriever

    @weave.op
    def predict(self, question: str) -> dict[str, Any]:
        return self._retriever.search(question)


__all__ = ["GoodMemRetrievalModel"]
