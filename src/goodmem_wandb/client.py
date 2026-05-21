"""GoodMem client for the goodmem_wandb integration.

Provides a Python client wrapping all GoodMem API operations:
List Embedders, List Spaces, Get Space, Create Space, Update Space,
Delete Space, Create Memory (text and file), List Memories,
Retrieve Memories, Get Memory, and Delete Memory.
"""

from __future__ import annotations

import base64
import json as _json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# Default chunking configuration matching the reference integration
_DEFAULT_CHUNK_SIZE = 256
_DEFAULT_CHUNK_OVERLAP = 25
_DEFAULT_KEEP_STRATEGY = "KEEP_END"
_DEFAULT_LENGTH_MEASUREMENT = "CHARACTER_COUNT"
_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Polling configuration
_DEFAULT_POLL_INTERVAL_S = 5
_DEFAULT_MAX_WAIT_S = 10

# HTTP request timeout (seconds)
_DEFAULT_TIMEOUT_S = 30

# MIME type mapping matching the reference integration
_MIME_TYPES: Dict[str, str] = {
    "pdf": "application/pdf",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "txt": "text/plain",
    "html": "text/html",
    "md": "text/markdown",
    "csv": "text/csv",
    "json": "application/json",
    "xml": "application/xml",
    "doc": "application/msword",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xls": "application/vnd.ms-excel",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "ppt": "application/vnd.ms-powerpoint",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


def _get_mime_type(extension: str) -> Optional[str]:
    """Return the MIME type for a file extension, or None if unknown."""
    ext = extension.lower().lstrip(".")
    return _MIME_TYPES.get(ext)


class GoodMemClient:
    """Client for interacting with the GoodMem API.

    Args:
        base_url: The base URL of the GoodMem API server
            (e.g., ``https://api.goodmem.ai`` or ``http://localhost:8080``).
        api_key: Your GoodMem API key for authentication (``X-API-Key``).
        max_wait_s: Maximum seconds to poll when waiting for indexing
            during retrieval. Defaults to 10.
        poll_interval_s: Seconds between poll attempts. Defaults to 5.
        verify_ssl: Whether to verify SSL certificates. Defaults to True.

    Example::

        from goodmem_wandb import GoodMemClient

        client = GoodMemClient(
            base_url="http://localhost:8080",
            api_key="gm_your_key_here",
        )

        # List available embedders
        embedders = client.list_embedders()

        # Create a space
        space = client.create_space(
            name="my-space",
            embedder_id=embedders[0]["embedderId"],
        )

        # Store a text memory
        memory = client.create_memory(
            space_id=space["spaceId"],
            text_content="Important information to remember.",
        )

        # Retrieve memories by semantic search
        results = client.retrieve_memories(
            query="important information",
            space_ids=[space["spaceId"]],
        )

        # Get a specific memory
        mem = client.get_memory(memory["memoryId"])

        # Delete a memory
        client.delete_memory(memory["memoryId"])
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        max_wait_s: int = _DEFAULT_MAX_WAIT_S,
        poll_interval_s: int = _DEFAULT_POLL_INTERVAL_S,
        verify_ssl: bool = True,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._max_wait_s = max_wait_s
        self._poll_interval_s = poll_interval_s
        self._verify_ssl = verify_ssl
        self._validate_connection()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {
            "X-API-Key": self._api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _validate_connection(self) -> None:
        """Validate the API key and base URL by listing spaces."""
        try:
            resp = requests.get(
                f"{self._base_url}/v1/spaces",
                headers=self._headers(),
                verify=self._verify_ssl,
                timeout=_DEFAULT_TIMEOUT_S,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise ConnectionError(
                f"Failed to connect to GoodMem at {self._base_url}. "
                f"Check your base URL and API key. Error: {exc}"
            ) from exc

    def _request(
        self,
        method: str,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        accept: Optional[str] = None,
    ) -> requests.Response:
        """Send an HTTP request to the GoodMem API.

        Raises a descriptive error on failure.
        """
        headers = self._headers()
        if accept:
            headers["Accept"] = accept

        try:
            resp = requests.request(
                method,
                f"{self._base_url}{path}",
                headers=headers,
                json=json,
                verify=self._verify_ssl,
                timeout=_DEFAULT_TIMEOUT_S,
            )
            resp.raise_for_status()
            return resp
        except requests.HTTPError as exc:
            detail = ""
            try:
                body = exc.response.json()
                detail = body.get("message", body.get("error", ""))
            except Exception:
                detail = exc.response.text[:500] if exc.response.text else ""
            raise RuntimeError(
                f"GoodMem API error ({exc.response.status_code}) on "
                f"{method} {path}: {detail or exc}"
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(
                f"GoodMem request failed on {method} {path}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # List Embedders
    # ------------------------------------------------------------------

    def list_embedders(self) -> List[Dict[str, Any]]:
        """List available embedder models.

        Returns:
            A list of embedder dicts, each containing at least
            ``embedderId`` and ``modelIdentifier``.
        """
        resp = self._request("GET", "/v1/embedders")
        body = resp.json()
        return body if isinstance(body, list) else body.get("embedders", [])

    # ------------------------------------------------------------------
    # List Spaces
    # ------------------------------------------------------------------

    def list_spaces(self) -> List[Dict[str, Any]]:
        """List all spaces.

        Returns:
            A list of space dicts, each containing at least
            ``spaceId`` and ``name``.
        """
        resp = self._request("GET", "/v1/spaces")
        body = resp.json()
        return body if isinstance(body, list) else body.get("spaces", [])

    # ------------------------------------------------------------------
    # Create Space
    # ------------------------------------------------------------------

    def create_space(
        self,
        name: str,
        embedder_id: str,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
        keep_strategy: str = _DEFAULT_KEEP_STRATEGY,
        length_measurement: str = _DEFAULT_LENGTH_MEASUREMENT,
    ) -> Dict[str, Any]:
        """Create a new space or reuse an existing one.

        A space is a logical container for organizing related memories,
        configured with embedders that convert text to vector embeddings.
        If a space with the given *name* already exists, its metadata is
        returned instead of creating a duplicate.

        Args:
            name: A unique name for the space.
            embedder_id: The embedder model ID (from :meth:`list_embedders`).
            chunk_size: Characters per chunk when splitting documents.
            chunk_overlap: Overlapping characters between consecutive chunks.
            keep_strategy: Where to attach the separator when splitting.
                One of ``KEEP_END``, ``KEEP_START``, or ``DISCARD``.
            length_measurement: How chunk size is measured.
                One of ``CHARACTER_COUNT`` or ``TOKEN_COUNT``.

        Returns:
            A dict with ``spaceId``, ``name``, ``embedderId``, ``reused``,
            and ``message``.
        """
        try:
            spaces = self.list_spaces()
            existing = next((s for s in spaces if s.get("name") == name), None)
            if existing:
                return {
                    "success": True,
                    "spaceId": existing["spaceId"],
                    "name": existing["name"],
                    "embedderId": embedder_id,
                    "message": "Space already exists, reusing existing space",
                    "reused": True,
                }
        except RuntimeError:
            logger.debug("Failed to list spaces for dedup check, proceeding to create")

        request_body = {
            "name": name,
            "spaceEmbedders": [
                {"embedderId": embedder_id, "defaultRetrievalWeight": 1.0},
            ],
            "defaultChunkingConfig": {
                "recursive": {
                    "chunkSize": chunk_size,
                    "chunkOverlap": chunk_overlap,
                    "separators": _DEFAULT_SEPARATORS,
                    "keepStrategy": keep_strategy,
                    "separatorIsRegex": False,
                    "lengthMeasurement": length_measurement,
                },
            },
        }

        resp = self._request("POST", "/v1/spaces", json=request_body)
        body = resp.json()
        return {
            "success": True,
            "spaceId": body["spaceId"],
            "name": body["name"],
            "embedderId": embedder_id,
            "chunkingConfig": request_body["defaultChunkingConfig"],
            "message": "Space created successfully",
            "reused": False,
        }

    # ------------------------------------------------------------------
    # Get Space
    # ------------------------------------------------------------------

    def get_space(self, space_id: str) -> Dict[str, Any]:
        """Fetch a single space by its ID.

        Args:
            space_id: The UUID of the space.

        Returns:
            A dict with ``success`` and ``space`` (the full space object
            including ``spaceId``, ``name``, ``labels``, ``spaceEmbedders``,
            and ``defaultChunkingConfig``).
        """
        if not space_id or not space_id.strip():
            raise ValueError("space_id is required and cannot be empty.")

        resp = self._request("GET", f"/v1/spaces/{space_id}")
        return {
            "success": True,
            "space": resp.json(),
        }

    # ------------------------------------------------------------------
    # Update Space
    # ------------------------------------------------------------------

    def update_space(
        self,
        space_id: str,
        name: Optional[str] = None,
        replace_labels: Optional[Dict[str, str]] = None,
        merge_labels: Optional[Dict[str, str]] = None,
        public_read: Optional[bool] = None,
        default_chunking_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update mutable fields on an existing space.

        Only the provided fields are sent in the request body; other
        space attributes (embedders, etc.) are left unchanged. The API
        distinguishes between replacing labels entirely and merging new
        labels into the existing set.

        Args:
            space_id: The UUID of the space to update.
            name: New name for the space.
            replace_labels: Labels dict that fully replaces the existing
                labels on the space (anything not listed is removed).
            merge_labels: Labels dict that is merged into the existing
                labels (existing keys not in the dict are preserved).
            public_read: Whether the space is publicly readable.
            default_chunking_config: Replacement chunking configuration,
                matching the structure used in :meth:`create_space`.

        Returns:
            A dict with ``success``, ``space`` (the updated space object),
            and ``message``.
        """
        if not space_id or not space_id.strip():
            raise ValueError("space_id is required and cannot be empty.")

        if replace_labels is not None and merge_labels is not None:
            raise ValueError(
                "Provide either replace_labels or merge_labels, not both."
            )

        update_body: Dict[str, Any] = {}
        if name is not None:
            update_body["name"] = name
        if replace_labels is not None:
            update_body["replaceLabels"] = replace_labels
        if merge_labels is not None:
            update_body["mergeLabels"] = merge_labels
        if public_read is not None:
            update_body["publicRead"] = public_read
        if default_chunking_config is not None:
            update_body["defaultChunkingConfig"] = default_chunking_config

        if not update_body:
            raise ValueError(
                "No fields to update. Provide at least one of: "
                "name, replace_labels, merge_labels, public_read, "
                "default_chunking_config."
            )

        resp = self._request("PUT", f"/v1/spaces/{space_id}", json=update_body)
        return {
            "success": True,
            "space": resp.json(),
            "message": "Space updated successfully",
        }

    # ------------------------------------------------------------------
    # Delete Space
    # ------------------------------------------------------------------

    def delete_space(self, space_id: str) -> Dict[str, Any]:
        """Permanently delete a space and all memories it contains.

        Args:
            space_id: The UUID of the space to delete.

        Returns:
            A dict with ``success``, ``spaceId``, and ``message``.
        """
        if not space_id or not space_id.strip():
            raise ValueError("space_id is required and cannot be empty.")

        self._request("DELETE", f"/v1/spaces/{space_id}")
        return {
            "success": True,
            "spaceId": space_id,
            "message": "Space deleted successfully",
        }

    # ------------------------------------------------------------------
    # Create Memory
    # ------------------------------------------------------------------

    def create_memory(
        self,
        space_id: str,
        text_content: Optional[str] = None,
        file_path: Optional[str] = None,
        source: Optional[str] = None,
        author: Optional[str] = None,
        tags: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Store a document or text as a new memory in a space.

        The memory is processed asynchronously -- chunked into searchable
        pieces and embedded into vectors.  Either *text_content* or
        *file_path* must be provided.  If both are given, the file takes
        priority.

        Args:
            space_id: The target space ID.
            text_content: Plain text content to store.
            file_path: Path to a local file (PDF, DOCX, image, etc.).
                Content type is auto-detected from the file extension.
            source: Where this memory came from (stored in metadata).
            author: Author of the content (stored in metadata).
            tags: Comma-separated tags (stored in metadata as an array).
            metadata: Additional key-value metadata (merged with above).

        Returns:
            A dict with ``memoryId``, ``spaceId``, ``status``, and
            ``contentType``.
        """
        if not space_id or not space_id.strip():
            raise ValueError("space_id is required and cannot be empty.")

        request_body: Dict[str, Any] = {"spaceId": space_id}

        if file_path:
            if not os.path.isfile(file_path):
                raise FileNotFoundError(
                    f"File not found: {file_path}. "
                    "Please provide a valid path to an existing file."
                )

            ext = os.path.splitext(file_path)[1]
            mime_type = _get_mime_type(ext) or "application/octet-stream"

            with open(file_path, "rb") as f:
                raw = f.read()

            if mime_type.startswith("text/"):
                request_body["contentType"] = mime_type
                request_body["originalContent"] = raw.decode("utf-8")
            else:
                request_body["contentType"] = mime_type
                request_body["originalContentB64"] = base64.b64encode(raw).decode(
                    "ascii"
                )
        elif text_content:
            request_body["contentType"] = "text/plain"
            request_body["originalContent"] = text_content
        else:
            raise ValueError(
                "No content provided. Please provide either text_content or file_path."
            )

        merged_metadata: Dict[str, Any] = {}
        if metadata and isinstance(metadata, dict):
            merged_metadata.update(metadata)
        if source:
            merged_metadata["source"] = source
        if author:
            merged_metadata["author"] = author
        if tags:
            merged_metadata["tags"] = [
                t.strip() for t in tags.split(",") if t.strip()
            ]
        if merged_metadata:
            request_body["metadata"] = merged_metadata

        resp = self._request("POST", "/v1/memories", json=request_body)
        body = resp.json()
        return {
            "success": True,
            "memoryId": body["memoryId"],
            "spaceId": body["spaceId"],
            "status": body.get("processingStatus", "PENDING"),
            "contentType": request_body["contentType"],
            "fileName": os.path.basename(file_path) if file_path else None,
            "message": "Memory created successfully",
        }

    # ------------------------------------------------------------------
    # List Memories
    # ------------------------------------------------------------------

    def list_memories(
        self,
        space_id: str,
        max_results: Optional[int] = None,
        next_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List memories stored in a space.

        The GoodMem API returns memories in pages; ``next_token`` can be
        passed back on a subsequent call to fetch the next page.

        Args:
            space_id: The UUID of the space whose memories to list.
            max_results: Optional page size. If omitted, the server default
                is used.
            next_token: Optional pagination cursor returned by a previous
                call as ``nextToken``.

        Returns:
            A dict with ``success``, ``memories`` (a list of memory objects),
            ``nextToken`` (``None`` if there are no further pages), and
            ``spaceId``.
        """
        if not space_id or not space_id.strip():
            raise ValueError("space_id is required and cannot be empty.")

        query_parts: List[str] = []
        if max_results is not None:
            query_parts.append(f"maxResults={int(max_results)}")
        if next_token:
            from urllib.parse import quote

            query_parts.append(f"nextToken={quote(next_token, safe='')}")

        path = f"/v1/spaces/{space_id}/memories"
        if query_parts:
            path += "?" + "&".join(query_parts)

        resp = self._request("GET", path)
        body = resp.json()
        memories = (
            body
            if isinstance(body, list)
            else body.get("memories", [])
        )
        return {
            "success": True,
            "spaceId": space_id,
            "memories": memories,
            "nextToken": body.get("nextToken") if isinstance(body, dict) else None,
        }

    # ------------------------------------------------------------------
    # Retrieve Memories
    # ------------------------------------------------------------------

    def retrieve_memories(
        self,
        query: str,
        space_ids: List[str],
        max_results: int = 5,
        include_memory_definition: bool = True,
        wait_for_indexing: bool = True,
        reranker_id: Optional[str] = None,
        llm_id: Optional[str] = None,
        relevance_threshold: Optional[float] = None,
        llm_temperature: Optional[float] = None,
        chronological_resort: bool = False,
    ) -> Dict[str, Any]:
        """Perform similarity-based semantic retrieval across spaces.

        Returns matching chunks ranked by relevance, with optional full
        memory definitions.

        Args:
            query: A natural language query for semantic search.
            space_ids: One or more space IDs to search across.
            max_results: Maximum number of results to return.
            include_memory_definition: Fetch full memory metadata alongside
                matched chunks.
            wait_for_indexing: Retry for up to ``max_wait_s`` seconds when
                no results are found (useful when memories were just added).
            reranker_id: Optional reranker model ID.
            llm_id: Optional LLM ID for contextual responses.
            relevance_threshold: Minimum score (0-1) for results.
                Only used with reranker or LLM.
            llm_temperature: Creativity setting (0-2) for LLM.
                Only used when ``llm_id`` is set.
            chronological_resort: Reorder results by creation time.

        Returns:
            A dict with ``results``, ``memories``, ``totalResults``,
            ``query``, and optionally ``abstractReply``.
        """
        if not space_ids:
            raise ValueError("At least one space must be selected.")

        space_keys = [{"spaceId": sid} for sid in space_ids]

        request_body: Dict[str, Any] = {
            "message": query,
            "spaceKeys": space_keys,
            "requestedSize": max_results,
            "fetchMemory": include_memory_definition,
        }

        if reranker_id or llm_id:
            config: Dict[str, Any] = {}
            if reranker_id:
                config["reranker_id"] = reranker_id
            if llm_id:
                config["llm_id"] = llm_id
            if relevance_threshold is not None:
                config["relevance_threshold"] = relevance_threshold
            if llm_temperature is not None:
                config["llm_temp"] = llm_temperature
            if max_results:
                config["max_results"] = max_results
            if chronological_resort:
                config["chronological_resort"] = True

            request_body["postProcessor"] = {
                "name": "com.goodmem.retrieval.postprocess.ChatPostProcessorFactory",
                "config": config,
            }

        start_time = time.time()
        last_result: Optional[Dict[str, Any]] = None

        while True:
            resp = self._request(
                "POST",
                "/v1/memories:retrieve",
                json=request_body,
                accept="application/x-ndjson",
            )

            results: List[Dict[str, Any]] = []
            memories: List[Dict[str, Any]] = []
            result_set_id = ""
            abstract_reply: Optional[Dict[str, Any]] = None

            response_text = resp.text
            for line in response_text.strip().split("\n"):
                json_str = line.strip()
                if not json_str:
                    continue
                if json_str.startswith("data:"):
                    json_str = json_str[5:].strip()
                if json_str.startswith("event:") or not json_str:
                    continue
                try:
                    item = _json.loads(json_str)
                    if item.get("resultSetBoundary"):
                        result_set_id = item["resultSetBoundary"].get(
                            "resultSetId", ""
                        )
                    elif item.get("memoryDefinition"):
                        memories.append(item["memoryDefinition"])
                    elif item.get("abstractReply"):
                        abstract_reply = item["abstractReply"]
                    elif item.get("retrievedItem"):
                        chunk_data = item["retrievedItem"].get("chunk", {})
                        chunk = chunk_data.get("chunk", {})
                        results.append(
                            {
                                "chunkId": chunk.get("chunkId"),
                                "chunkText": chunk.get("chunkText"),
                                "memoryId": chunk.get("memoryId"),
                                "relevanceScore": chunk_data.get("relevanceScore"),
                                "memoryIndex": chunk_data.get("memoryIndex"),
                            }
                        )
                except _json.JSONDecodeError:
                    logger.debug("Skipping non-JSON line in retrieve response: %s", json_str[:100])

            last_result = {
                "success": True,
                "resultSetId": result_set_id,
                "results": results,
                "memories": memories,
                "totalResults": len(results),
                "query": query,
            }
            if abstract_reply:
                last_result["abstractReply"] = abstract_reply

            if results or not wait_for_indexing:
                return last_result

            elapsed = time.time() - start_time
            if elapsed >= self._max_wait_s:
                last_result["message"] = (
                    f"No results found after waiting {self._max_wait_s} seconds "
                    "for indexing. Memories may still be processing."
                )
                return last_result

            time.sleep(self._poll_interval_s)

    # ------------------------------------------------------------------
    # Get Memory
    # ------------------------------------------------------------------

    def get_memory(
        self,
        memory_id: str,
        include_content: bool = True,
    ) -> Dict[str, Any]:
        """Fetch a specific memory record by its ID.

        Args:
            memory_id: The UUID of the memory.
            include_content: Whether to also fetch the original document
                content.

        Returns:
            A dict with ``memory`` and optionally ``content``.
        """
        if not memory_id or not memory_id.strip():
            raise ValueError("memory_id is required and cannot be empty.")

        resp = self._request("GET", f"/v1/memories/{memory_id}")
        result: Dict[str, Any] = {
            "success": True,
            "memory": resp.json(),
        }

        if include_content:
            try:
                content_resp = self._request(
                    "GET", f"/v1/memories/{memory_id}/content"
                )
                content_type = content_resp.headers.get("Content-Type", "")
                if "application/json" in content_type.lower():
                    result["content"] = content_resp.json()
                elif content_type.startswith("text/") or not content_type:
                    result["content"] = content_resp.text
                    result["contentType"] = content_type or "text/plain"
                else:
                    result["contentB64"] = base64.b64encode(
                        content_resp.content
                    ).decode("ascii")
                    result["contentType"] = content_type
            except RuntimeError as exc:
                logger.warning("Failed to fetch content for memory %s: %s", memory_id, exc)
                result["contentError"] = f"Failed to fetch content: {exc}"

        return result

    # ------------------------------------------------------------------
    # Delete Memory
    # ------------------------------------------------------------------

    def delete_memory(self, memory_id: str) -> Dict[str, Any]:
        """Permanently delete a memory and its associated chunks and embeddings.

        Args:
            memory_id: The UUID of the memory to delete.

        Returns:
            A dict with ``memoryId`` and ``message``.
        """
        if not memory_id or not memory_id.strip():
            raise ValueError("memory_id is required and cannot be empty.")

        self._request("DELETE", f"/v1/memories/{memory_id}")
        return {
            "success": True,
            "memoryId": memory_id,
            "message": "Memory deleted successfully",
        }
