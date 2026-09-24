"""Resolving a space by name.

Attaching by name is idempotent reuse: an existing space with the requested
embedder is returned as-is, and one built on a different embedder is an error
rather than a silent mismatch. Retrieval against the wrong embedder returns
plausible-looking nonsense, so it must never happen quietly.

This matches the ActivePieces connector's behaviour. GoodMem does not enforce
unique space names, so an ambiguous name is also an error.
"""

from __future__ import annotations

from typing import Any

from goodmem_wandb._typing import GoodmemClient

# A name is not unique, so the lookup is bounded rather than unbounded.
MAX_CANDIDATES = 200


class GoodMemSpaceError(RuntimeError):
    """A space could not be resolved unambiguously."""


def find_by_name(client: GoodmemClient, name: str) -> list[Any]:
    """Return every space named exactly ``name``.

    ``name_filter`` is a server-side substring match, so each candidate is
    re-checked for an exact name. Iterating the SDK's ``Page`` follows
    pagination for us; ``max_items`` bounds it.
    """
    page = client.spaces.list(name_filter=name, max_items=MAX_CANDIDATES)
    return [s for s in page if getattr(s, "name", None) == name]


def embedder_ids(space: Any) -> list[str]:
    return [
        e.embedder_id
        for e in (getattr(space, "space_embedders", None) or [])
        if getattr(e, "embedder_id", None)
    ]


def resolve(
    client: GoodmemClient,
    *,
    name: str,
    embedder_id: str | None = None,
    create: bool = False,
    chunking_config: dict[str, Any] | None = None,
) -> str:
    """Return the id of the space called ``name``.

    With ``create=True`` a missing space is created. An existing space is
    reused only when ``embedder_id`` is unset or already configured on it.
    """
    matches = find_by_name(client, name)
    if len(matches) > 1:
        raise GoodMemSpaceError(
            f"{len(matches)} spaces are named {name!r}. Pass space_id to choose "
            "one; GoodMem does not require space names to be unique."
        )

    if matches:
        space = matches[0]
        configured = embedder_ids(space)
        if embedder_id and embedder_id not in configured:
            raise GoodMemSpaceError(
                f"Space {name!r} ({space.space_id}) uses embedder(s) "
                f"{', '.join(configured) or 'none'}, not {embedder_id}. "
                "Retrieval across mismatched embedders returns meaningless "
                "results. Pick the existing embedder or use another name."
            )
        return str(space.space_id)

    if not create:
        raise GoodMemSpaceError(
            f"No space named {name!r}. Pass space_id, or set create_space=True "
            "to create it."
        )
    if not embedder_id:
        raise GoodMemSpaceError(
            f"Cannot create space {name!r} without embedder_id. List the "
            "available embedders with the GoodMem SDK or the goodmem CLI."
        )

    request: dict[str, Any] = {
        "name": name,
        "space_embedders": [
            {"embedderId": embedder_id, "defaultRetrievalWeight": 1.0}
        ],
        # The server rejects a create without one.
        "default_chunking_config": chunking_config or _DEFAULT_CHUNKING,
    }
    try:
        space = client.spaces.create(**request)
    except Exception as exc:
        # Another process may have created it between the lookup and the
        # create. Re-resolve rather than fail the caller.
        if not _is_conflict(exc):
            raise
        again = find_by_name(client, name)
        if len(again) == 1:
            return str(again[0].space_id)
        raise
    return str(space.space_id)


_DEFAULT_CHUNKING: dict[str, Any] = {
    "recursive": {
        "chunkSize": 256,
        "chunkOverlap": 25,
        "separators": ["\n\n", "\n", ". ", " ", ""],
        "keepStrategy": "KEEP_END",
        "separatorIsRegex": False,
        "lengthMeasurement": "CHARACTER_COUNT",
    }
}


def _is_conflict(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    return status == 409 or "conflict" in type(exc).__name__.lower()
