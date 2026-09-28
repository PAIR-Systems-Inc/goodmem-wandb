# Changelog

## 0.2.2

Two retrieval fixes. Both were reproduced against a live GoodMem server
(v1.0.320) before the fix and re-measured after it.

### Fixed

* **A failed reranker's hits were labelled and thresholded as reranker
  scores.** With `reranker_id` set and the reranker failing, the server still
  returns the vector-stage hits and reports `RERANKING_FAILED` (plus
  `NOT_FOUND` for a missing reranker). 0.2.1 labelled those hits
  `score_kind: "reranker"` from configuration and applied `min_score` to
  their vector scores: live, a missing reranker with `min_score=0.0` returned
  0 of 3 hits and a warning about the reranker's scale. `score_kind` is now
  decided from the whole response, so those hits are `"vector"`, `min_score`
  is not applied to them, and all 3 come back with `partial: True` and both
  statuses. A working reranker and any unrelated status are unaffected.
* **Boolean and float `metadata_filter` values matched nothing.** Every
  value was compared as text, so `{"flag": True}` was sent as
  `CAST(val('$.flag') AS TEXT) = 'True'` and `{"n": 5.0}` as `= '5.0'`; the
  server accepts both and returns 0 results. Each value is now compared as
  its own type: `str` as `TEXT` (escaping unchanged), `bool` as `BOOLEAN`,
  `int`/`float` as `NUMERIC` (finite, written as a plain decimal). `None` and
  other types raise `ValueError` before any request instead of being sent as
  the text `'None'`.

### Behaviour change

* A Python number or boolean in `metadata_filter` is now compared as a number
  or a boolean. To compare text, pass a string: `{"year": "2026"}`. Measured
  live, the typed casts also accept string-stored values the server can
  convert (a stored `"5"` matches `{"n": 5}`; a stored `"true"` matches
  `{"flag": True}`), and a value it cannot convert simply does not match.

### Tests

* 86 offline tests (was 44): `tests/test_reranker_fallback.py` (16) replays
  the live-captured degraded streams, `tests/test_typed_filters.py` (26)
  checks the filter each value sends through the real SDK. 31 of the 42 fail
  on 0.2.1; the rest are controls.
* 20 live tests (was 11): the failed-reranker fallback and 8 typed-filter
  cases.

## 0.2.1

Documentation only; no behaviour change in the package.

### Fixed

* **The README's evaluation example never evaluated anything.** weave's
  `Evaluation.evaluate` is a coroutine, and the README, `examples/demo.py`
  and the `GoodMemRetrievalModel` docstring called it bare: each call
  returned an un-awaited coroutine, made no GoodMem request and logged
  nothing to Weave. All three now use `asyncio.run(evaluation.evaluate(...))`.
* The `search()` return table in the README now lists `query`, and the live
  test notes say one test (not several) needs `GOODMEM_RERANKER_ID`.

### Added

* `tests/test_readme.py` runs the README's Python blocks as written, the
  demo script and the docstring example against the captured fixtures, and
  checks the documented return keys against a real `search()` result.
  44 offline tests (was 39); the 11 live tests are unchanged.

## 0.2.0

A rewrite. 0.1.0 shipped no Weave integration at all: it was a hand-rolled
`requests` client with "wandb" in the package name. 0.2.0 is what the name
promises, built on the official `goodmem` SDK.

Every claim below was reproduced against a live GoodMem server (v1.0.320)
before the fix, and the offline tests replay bytes captured from it.

### Security

* **The API key is no longer a serializable field.** Weave publishes a
  `weave.Object`'s pydantic fields verbatim to the W&B trace server, and its
  `should_redact` helper does not run on that path — a field named `api_key`
  on a `weave.Model` is uploaded in plaintext the first time an evaluation
  runs. Credentials live on a private connection instead, asserted by
  `test_credentials_are_not_weave_fields` and
  `test_a_field_literally_named_api_key_would_leak`.
* **Removed the hardcoded API key** from the test suite. 0.1.0 shipped a real
  key as the default value of `GOODMEM_API_KEY` in `tests/test_live.py`, on a
  public branch. It is now required from the environment with no fallback, and
  the key itself has been purged from this repository's history.
* **Removed the `~/Downloads/*.pdf` default.** The 0.1.0 test picked an
  arbitrary PDF out of the user's Downloads folder and uploaded it to the
  server.

### Fixed

| Behaviour | 0.1.0 | 0.2.0 |
| --- | --- | --- |
| `status` events in the retrieval stream | No branch for them — dropped entirely | Classified: `FEATURE_DISABLED / summarization` is informational, a real status marks the result `partial`, and a code the SDK does not recognise is surfaced as `UNKNOWN` rather than assumed harmless |
| A retrieval that failed | `success: true, totalResults: 0`, blaming indexing | Empty `hits` with `partial: true` and the server's `statuses` — distinguishable from a miss, never raised |
| `relevance_threshold` | Forwarded to the server; a documented value of `0.5` filtered out a reranked hit scoring `0.4224`, then blamed indexing | `min_score`, applied client-side and only with a reranker configured |
| Relevance scores | Documented as "0–1, higher is better" | Reported as the server sends them, in the server's order, with `score_kind` on each hit. Both scales go negative |
| `create_space` on an existing name | Returned `reused: true` and echoed back *your* embedder id, not the space's | Reuses only when the embedder matches; a mismatch or an ambiguous name is an error |
| `list_spaces` for the reuse check | Sent no page parameters and dropped `nextToken`, so only page 1 was ever checked | Server-side `name_filter` plus the SDK's pagination, with every candidate re-checked for an exact name |
| `update_space(public_read=…)` | Advertised as "flip visibility"; the server removed the field and returns HTTP 400 | Removed |
| A non-UTF-8 text file | `UnicodeDecodeError` before the request was built | Not applicable — ingestion is out of scope for this package; use the `goodmem` SDK |
| Constructing a client | Did a full `GET /v1/spaces`, surfacing a 401 as `ConnectionError` | No network call; errors surface from the operation that caused them |
| Metadata filtering | Not supported at all | `filter` and `metadata_filter`, safely quoted for the live filter grammar |
| Retry-on-empty | Re-POSTed the whole retrieval every 5s for 10s, re-invoking the LLM each time | Removed; an empty result is reported as empty |

### Added

* `GoodMemRetriever` — a `weave.Object` whose `search()` is a `weave.op`.
* `GoodMemRetrievalModel` — a `weave.Model` for `weave.Evaluation`.
* `RecallAtK`, `MRR`, `FactRecall`, `RetrievalHealth` — retrieval scorers.
* `trace_chunk_text=False` for corpora that should not be uploaded to W&B.
* 39 offline tests over captured server bytes, and 11 live tests with verified
  teardown. 0.1.0 had one live-only script that CI never ran.
* CI now runs lint, type-check and the offline suite. 0.1.0's CI only built a
  wheel and imported it.

### Migration

`GoodMemClient` is gone. It was a generic REST client; the `goodmem` SDK does
the same job better:

```python
# 0.1.0
from goodmem_wandb import GoodMemClient
client = GoodMemClient(base_url=..., api_key=...)
results = client.retrieve_memories(query="…", space_ids=[space_id])

# 0.2.0 — for tracing and evaluation
from goodmem_wandb import GoodMemRetriever
result = GoodMemRetriever(space_id=space_id).search("…")

# 0.2.0 — for plain API access, use the SDK directly
from goodmem import Goodmem
Goodmem(base_url=..., api_key=...).memories.retrieve(...)
```

| 0.1.0 | 0.2.0 |
| --- | --- |
| `retrieve_memories(query, space_ids, max_results)` | `GoodMemRetriever(space_ids=…, limit=…).search(query)` |
| `results[i]["relevanceScore"]` | `result["hits"][i]["score"]` + `["score_kind"]` |
| `results[i]["chunkText"]` | `result["hits"][i]["chunk_text"]` |
| `relevance_threshold=…` | `min_score=…`, with `reranker_id` set |
| `reranker_id` / `llm_id` / `llm_temperature` | same names on the retriever |
| `create_space` / `get_space` / `update_space` / `delete_space` | `goodmem.Goodmem().spaces.*` |
| `create_memory` / `list_memories` / `get_memory` / `delete_memory` | `goodmem.Goodmem().memories.*` |

Python 3.9 is no longer supported; 3.10+ is required.

## 0.1.0

Initial release.
