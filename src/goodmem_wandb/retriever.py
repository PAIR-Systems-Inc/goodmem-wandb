"""A GoodMem retriever that records what it did into Weave."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import Field, PrivateAttr
import weave

from goodmem_wandb._connection import GoodMemConnection, split_connection_kwargs
from goodmem_wandb._results import (
    GoodMemRetrievalError,
    abstract_reply,
    classify,
    hits_from_events,
)
from goodmem_wandb._spaces import GoodMemSpaceError, resolve
from goodmem_wandb.filters import combine, from_mapping


class GoodMemRetriever(weave.Object):
    """Retrieve from GoodMem spaces, traced as a Weave op.

    Every :meth:`search` call becomes a Weave call whose output records the
    hits, the score *and the kind of score*, any server statuses, and whether
    the result is partial. A degraded retrieval is therefore visible in the
    Weave UI instead of looking like a thin day::

        import weave
        from goodmem_wandb import GoodMemRetriever

        weave.init("my-project")
        retriever = GoodMemRetriever(space_name="docs")   # credentials from env
        result = retriever.search("how do I rotate a key?")

    Credentials are never weave fields. Weave publishes an object's pydantic
    fields verbatim to the trace server, so ``base_url``/``api_key`` are
    accepted as constructor keywords and kept on a private connection. What is
    published is the retrieval configuration, which is what you want versioned.

    Scores are reported exactly as the server returns them, in the server's
    order. A GoodMem vector score is an opaque similarity that is frequently
    negative and whose best match may be the *lowest* number; a reranker score
    is a different scale that can also be negative. They are never mixed,
    re-sorted or thresholded against each other -- ``score_kind`` on each hit
    says which one you are looking at, and ``min_score`` is only honoured with
    a reranker configured.
    """

    space_id: str | None = None
    space_ids: list[str] = Field(default_factory=list)
    space_name: str | None = Field(
        default=None,
        description=(
            "Attach by name. An existing space is reused when its embedder "
            "matches embedder_id; a different embedder is an error."
        ),
    )
    embedder_id: str | None = None
    create_space: bool = False

    limit: int = Field(default=5, gt=0)
    fetch_k: int | None = Field(
        default=None,
        gt=0,
        description="Candidates to retrieve before reranking. Defaults to limit.",
    )
    reranker_id: str | None = None
    min_score: float | None = Field(
        default=None,
        description="Only applied when reranker_id is set. See the class docstring.",
    )
    filter: str | None = Field(
        default=None,
        description="A GoodMem filter expression applied to every space searched.",
    )
    metadata_filter: dict[str, Any] | None = Field(
        default=None,
        description="Field/value pairs, safely quoted and AND-ed into the filter.",
    )

    llm_id: str | None = Field(
        default=None,
        description="Enables the server's abstract reply. Costs an LLM call per search.",
    )
    llm_temperature: float | None = None

    trace_chunk_text: bool = Field(
        default=True,
        description=(
            "Include retrieved text in the Weave trace. Set False for a "
            "sensitive corpus: chunk ids, scores and statuses are still traced."
        ),
    )

    _conn: GoodMemConnection = PrivateAttr()
    _resolved_space_id: str | None = PrivateAttr(default=None)

    def __init__(self, **data: Any) -> None:
        conn = split_connection_kwargs(data)
        super().__init__(**data)
        self._conn = conn
        if not (self.space_id or self.space_ids or self.space_name):
            raise ValueError(
                "Provide space_id, space_ids or space_name so the retriever "
                "knows what to search."
            )

    # ------------------------------------------------------------- internals
    def _targets(self, client: Any) -> list[str]:
        ids = list(self.space_ids)
        if self.space_id:
            ids.insert(0, self.space_id)
        if self.space_name:
            if self._resolved_space_id is None:
                self._resolved_space_id = resolve(
                    client,
                    name=self.space_name,
                    embedder_id=self.embedder_id,
                    create=self.create_space,
                )
            ids.insert(0, self._resolved_space_id)
        # Preserve order, drop repeats.
        return list(dict.fromkeys(ids))

    def _expression(self, extra: dict[str, Any] | None) -> str | None:
        return combine(
            self.filter,
            from_mapping(self.metadata_filter) if self.metadata_filter else None,
            from_mapping(extra) if extra else None,
        )

    # --------------------------------------------------------------- traced
    @weave.op
    def search(
        self,
        query: str,
        *,
        limit: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Retrieve for one query.

        Returns ``{"query", "hits", "statuses", "partial", "abstract_reply",
        "space_ids", "score_kind"}``. ``partial`` is True when the server
        reported a real problem but still returned usable hits; a retrieval
        that reported a problem and returned nothing raises
        :class:`~goodmem_wandb._results.GoodMemRetrievalError` rather than
        returning an empty list that reads as "no matches".
        """
        if not query or not query.strip():
            raise ValueError("query cannot be empty.")

        want = limit if limit is not None else self.limit
        reranked = bool(self.reranker_id)
        expression = self._expression(metadata_filter)

        with self._conn.session() as client:
            targets = self._targets(client)
            kwargs: dict[str, Any] = {
                "message": query,
                "requested_size": self.fetch_k or want,
                "fetch_memory": True,
                "stream": False,
            }
            if expression is None:
                kwargs["space_ids"] = targets
            else:
                kwargs["space_keys"] = [
                    {"spaceId": sid, "filter": expression} for sid in targets
                ]
            if reranked:
                kwargs["reranker_id"] = self.reranker_id
                kwargs["max_results"] = want
            if self.llm_id:
                kwargs["llm_id"] = self.llm_id
                if self.llm_temperature is not None:
                    kwargs["llm_temp"] = self.llm_temperature

            events = list(client.memories.retrieve(**kwargs))

        statuses, degraded = classify(events)
        hits = hits_from_events(events, reranked=reranked)

        if degraded and not hits:
            # A failed retrieval must not be indistinguishable from an empty one.
            raise GoodMemRetrievalError(
                "; ".join(
                    f"{s.get('code', 'UNKNOWN')}: {s.get('message', '')}"
                    for s in statuses
                )
                or "Retrieval failed",
                statuses=statuses,
            )

        if reranked and self.min_score is not None:
            hits = [
                h for h in hits if h["score"] is None or h["score"] >= self.min_score
            ]
        # Server order is authoritative and is not re-sorted here.
        hits = hits[:want]

        if not self.trace_chunk_text:
            hits = [{**h, "chunk_text": None} for h in hits]

        return {
            "query": query,
            "hits": hits,
            "score_kind": "reranker" if reranked else "vector",
            "statuses": statuses,
            "partial": bool(degraded and hits),
            "abstract_reply": abstract_reply(events),
            "space_ids": targets,
        }

    @weave.op
    def search_many(self, queries: Sequence[str]) -> list[dict[str, Any]]:
        """Retrieve for several queries. Each one is its own traced child call."""
        return [self.search(q) for q in queries]

    # ------------------------------------------------------------ convenience
    def as_context(self, query: str, *, separator: str = "\n\n---\n\n") -> str:
        """Retrieve and join the hits into a prompt-ready block."""
        result = self.search(query)
        return separator.join(
            h["chunk_text"] for h in result["hits"] if h.get("chunk_text")
        )


__all__ = ["GoodMemRetrievalError", "GoodMemRetriever", "GoodMemSpaceError"]
