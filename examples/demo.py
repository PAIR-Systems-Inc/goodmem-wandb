"""Minimal demo for goodmem-wandb.

Creates a space, stores a text memory and a PDF, semantically retrieves
the content, and leaves the resources on the server for inspection.

Configure with the same env vars as ``tests/test_live.py``.
"""

from __future__ import annotations

import os
import time

from goodmem_wandb import GoodMemClient

BASE_URL = os.environ.get("GOODMEM_BASE_URL", "https://localhost:8080")
API_KEY = os.environ.get("GOODMEM_API_KEY", "gm_g5xcse2tjgcznlg45c5le4ti5q")
EMBEDDER_ID = os.environ.get(
    "GOODMEM_EMBEDDER_ID", "019cfd1c-c033-7517-b7de-f73941a0464b"
)
SPACE_NAME = f"goodmem-wandb-demo-{int(time.time())}"


def main() -> None:
    client = GoodMemClient(base_url=BASE_URL, api_key=API_KEY, verify_ssl=False)

    print("→ list_embedders")
    embedders = client.list_embedders()
    print(f"  found {len(embedders)} embedder(s)")

    print(f"→ create_space '{SPACE_NAME}'")
    space = client.create_space(name=SPACE_NAME, embedder_id=EMBEDDER_ID)
    space_id = space["spaceId"]
    print(f"  spaceId = {space_id}")

    print("→ create_memory (text)")
    mem = client.create_memory(
        space_id=space_id,
        text_content="GoodMem is a memory layer for AI agents.",
        source="goodmem-wandb-demo",
    )
    print(f"  memoryId = {mem['memoryId']}")

    print("→ retrieve_memories")
    out = client.retrieve_memories(
        query="memory layer for AI agents",
        space_ids=[space_id],
        max_results=3,
    )
    print(f"  totalResults = {out['totalResults']}")
    for r in out["results"][:3]:
        snippet = (r.get("chunkText") or "")[:80].replace("\n", " ")
        print(f"    • {snippet}")

    print("\nResources left on server:")
    print(f"  spaceId  = {space_id}")
    print(f"  memoryId = {mem['memoryId']}")


if __name__ == "__main__":
    main()
