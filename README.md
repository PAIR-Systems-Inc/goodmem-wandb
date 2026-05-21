# goodmem-wandb

GoodMem client packaged for wandb agent integrations. GoodMem is a memory
layer for AI agents with support for semantic storage, retrieval, and
summarization. This package exposes the full GoodMem API surface as a clean
Python client that can be used with any wandb agent (or any Python app).

## Install

```bash
pip install goodmem-wandb
```

## Quick start

```python
from goodmem_wandb import GoodMemClient

client = GoodMemClient(
    base_url="http://localhost:8080",
    api_key="gm_your_key_here",
)

# 1. List available embedders
embedders = client.list_embedders()

# 2. Create a space
space = client.create_space(
    name="my-space",
    embedder_id=embedders[0]["embedderId"],
)

# 3. Store a text memory
memory = client.create_memory(
    space_id=space["spaceId"],
    text_content="Important information to remember.",
    source="manual",
    tags="test,demo",
)

# 4. Store a PDF memory
pdf_memory = client.create_memory(
    space_id=space["spaceId"],
    file_path="/path/to/document.pdf",
)

# 5. Retrieve memories by semantic search
results = client.retrieve_memories(
    query="important information",
    space_ids=[space["spaceId"]],
    reranker_id="<reranker-uuid>",     # optional
    llm_id="<llm-uuid>",               # optional, enables abstractReply
    relevance_threshold=0.5,           # optional
    llm_temperature=0.2,               # optional
    max_results=5,
    chronological_resort=False,
)

# 6. Inspect / delete
mem = client.get_memory(memory["memoryId"])
client.delete_memory(memory["memoryId"])
```

## Available operations

| Operation         | Method                                      | Purpose                                       |
| ----------------- | ------------------------------------------- | --------------------------------------------- |
| List Embedders    | `list_embedders()`                          | Discover available embedder models            |
| List Spaces       | `list_spaces()`                             | List all spaces                               |
| Get Space         | `get_space(space_id)`                       | Fetch a single space                          |
| Create Space      | `create_space(name, embedder_id, ...)`      | Create or reuse a space                       |
| Update Space      | `update_space(space_id, ...)`               | Rename, relabel, or flip visibility           |
| Delete Space      | `delete_space(space_id)`                    | Delete a space and its memories               |
| Create Memory     | `create_memory(space_id, ...)`              | Store text or a file as a memory              |
| List Memories     | `list_memories(space_id, ...)`              | List memories in a space (paginated)          |
| Retrieve Memories | `retrieve_memories(query, space_ids, ...)`  | Semantic search across spaces                 |
| Get Memory        | `get_memory(memory_id)`                     | Fetch one memory and its content              |
| Delete Memory     | `delete_memory(memory_id)`                  | Delete a memory                               |

### Retrieval options

`retrieve_memories` supports the full GoodMem post-processor configuration:

| Argument               | Type    | Purpose                                                            |
| ---------------------- | ------- | ------------------------------------------------------------------ |
| `reranker_id`          | str     | UUID of a reranker model to improve result ordering                |
| `llm_id`               | str     | UUID of an LLM to generate contextual responses (`abstractReply`)  |
| `relevance_threshold`  | float   | Minimum score (0–1) for including results                          |
| `llm_temperature`      | float   | Creativity setting for LLM generation (0–2)                        |
| `max_results`          | int     | Limit the number of returned memories                              |
| `chronological_resort` | bool    | Reorder results by creation time                                   |

## Authentication

The `GoodMemClient` requires:

- **base_url**: The base URL of your GoodMem instance (e.g.
  `http://localhost:8080`, `https://api.goodmem.ai`).
- **api_key**: Your GoodMem API key (`X-API-Key`, starts with `gm_`).

For self-signed certs (e.g. local dev with `https://localhost:8080`), pass
`verify_ssl=False`.

## License

MIT
