# goodmem-wandb

GoodMem retrieval for [Weights & Biases Weave](https://weave-docs.wandb.ai/).

Every retrieval is a traced Weave op, so a RAG call shows up in the Weave UI
with its query, its latency, each hit's score **and what kind of score that
is**, and any degradation the server reported. The same retriever can be
wrapped as a `weave.Model` and measured with `weave.Evaluation`.

```bash
pip install goodmem-wandb
```

## Trace a retrieval

```python
import weave
from goodmem_wandb import GoodMemRetriever

weave.init("my-project")

retriever = GoodMemRetriever(space_name="docs")   # credentials from the environment
result = retriever.search("how do I rotate an API key?")

for hit in result["hits"]:
    print(hit["score"], hit["score_kind"], hit["chunk_text"][:80])
```

`search()` returns:

| Key | What it is |
| --- | --- |
| `hits` | `chunk_id`, `chunk_text`, `memory_id`, `space_id`, `source`, `score`, `score_kind`, `metadata` — in the server's order |
| `score_kind` | `"vector"` or `"reranker"`. They are different scales; see below |
| `statuses` | Server statuses that indicate a real problem, `[]` when clean |
| `partial` | `True` when the server reported a problem *and* still returned usable hits |
| `abstract_reply` | The server-generated summary, only when `llm_id` is set |
| `space_ids` | Which spaces were actually searched |

Credentials come from `GOODMEM_BASE_URL` and `GOODMEM_API_KEY`, or as
constructor keywords:

```python
GoodMemRetriever(space_name="docs", base_url="https://localhost:8080",
                 api_key="gm_…", verify_ssl=False)
```

They are deliberately **not** Weave fields. Weave publishes an object's
pydantic fields verbatim to the trace server and its redaction helper does not
run on that path, so a field named `api_key` on a `weave.Model` would be
uploaded to W&B in plaintext. This package keeps the connection on a private
attribute; `tests/test_regressions.py` asserts it.

## Evaluate a retrieval configuration

```python
import weave
from goodmem_wandb import GoodMemRetrievalModel, RecallAtK, MRR, FactRecall, RetrievalHealth

weave.init("my-project")

dataset = [
    {"question": "how do I rotate an API key?",
     "expected_memory_ids": ["01a0…"],
     "expected_text": "rotate the key from the console"},
]

baseline = GoodMemRetrievalModel(space_name="docs", limit=5)
reranked = GoodMemRetrievalModel(space_name="docs", limit=5, reranker_id="…")

evaluation = weave.Evaluation(
    dataset=dataset,
    scorers=[RecallAtK(k=5), MRR(), FactRecall(), RetrievalHealth()],
)
evaluation.evaluate(baseline)
evaluation.evaluate(reranked)   # compare the two in the Weave UI
```

Changing any field on the model versions it, so the two runs are directly
comparable. The credentials are not fields, so they are not part of the
version either.

| Scorer | Measures |
| --- | --- |
| `RecallAtK(k=5)` | Fraction of `expected_memory_ids` in the top *k* **distinct memories** (several chunks of one memory are one document) |
| `MRR()` | Reciprocal rank of the first expected memory; `0.0` if none was retrieved |
| `FactRecall()` | Whether a known fact actually appears in the retrieved text — survives re-chunking and re-embedding, unlike an id metric |
| `RetrievalHealth()` | Whether the retrieval was complete, so a silently-degraded run is visible as its own metric rather than only as a recall drop |

None of them score on the raw relevance number, because that number does not
mean the same thing between two configurations.

## Scores

GoodMem returns two different things in the same field, and this matters:

| | Range observed on a live server | Best match is |
| --- | --- | --- |
| Vector score | negative, e.g. `-0.6154 … -0.3873` | the **lowest** number |
| Reranker score | `0.2105 … -0.1081` — also goes negative | the **highest** number |

Those are real numbers from one capture over the same three memories. So:

* results keep **the server's order** and are never re-sorted here;
* `score_kind` on every hit says which scale you are looking at;
* `min_score` is only applied when `reranker_id` is set, and is applied
  client-side where you can see it, never sent as the server's
  `relevance_threshold`.

## Filtering

```python
retriever = GoodMemRetriever(space_name="docs", metadata_filter={"category": "billing"})
retriever.search("refunds", metadata_filter={"lang": "en"})   # AND-ed per call
```

Values are quoted for the GoodMem filter grammar (backslash escaping, verified
against a live server; control characters are refused rather than mangled).
For anything more complex, pass an expression directly:

```python
GoodMemRetriever(space_name="docs",
                 filter="CAST(val('$.year') AS TEXT) = '2026'")
```

## Attaching to a space by name

```python
GoodMemRetriever(space_name="docs", embedder_id="…")                    # reuse or fail
GoodMemRetriever(space_name="docs", embedder_id="…", create_space=True) # or create it
```

Attach-by-name is idempotent reuse: an existing space whose embedder matches is
reused; one built on a **different** embedder is an error, because retrieving
across mismatched embedders returns plausible-looking nonsense. A name that
matches more than one space is also an error — GoodMem does not require space
names to be unique. This matches the ActivePieces connector.

## Sensitive corpora

`trace_chunk_text=False` keeps chunk ids, scores and statuses in the trace but
leaves the retrieved text out of it.

## Development

```bash
pip install -e ".[dev]"
ruff check src tests && mypy && pytest -m "not integration"
```

The offline suite replays NDJSON captured from a live GoodMem server
(v1.0.320) through the real SDK decoders, so the wire format is never
invented. The live suite needs a server and is skipped without one:

```bash
GOODMEM_BASE_URL=… GOODMEM_API_KEY=… GOODMEM_EMBEDDER_ID=… \
  pytest -m integration
```

There is no default credential anywhere in this repository.

## License

MIT
