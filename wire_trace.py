"""Live wire trace for goodmem-wandb 0.2.0.

Two wires are recorded:
  * every HTTP request the GoodMem SDK makes, and
  * every payload the Weave client would upload to wandb.ai.

The second one is the point of this integration, and is where 0.1.0's
successor design would have leaked the API key.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from goodmem import Goodmem
import weave
from weave.trace import weave_client
from weave.trace.context import weave_client_context

from goodmem_wandb import (
    MRR,
    FactRecall,
    GoodMemRetrievalModel,
    GoodMemRetriever,
    GoodMemSpaceError,
    RecallAtK,
    RetrievalHealth,
)

BASE_URL = os.environ["GOODMEM_BASE_URL"]
API_KEY = os.environ["GOODMEM_API_KEY"]
EMBEDDER = os.environ["GOODMEM_EMBEDDER_ID"]
RERANKER = os.environ.get("GOODMEM_RERANKER_ID", "")
SECRET_MARKER = API_KEY

W = 100
_n = 0
_spaces: list[str] = []


def banner(title: str) -> None:
    print("\n" + "=" * W)
    print(title)
    print("=" * W)


def clip(text: str, limit: int = 380) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… (+{len(text) - limit} bytes)"


class TracingTransport(httpx.BaseTransport):
    """Log every GoodMem request/response exactly as it goes over the wire."""

    def __init__(self) -> None:
        self._inner = httpx.HTTPTransport(verify=False)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        global _n
        _n += 1
        print(f"\n  ──▶ #{_n}  {request.method} {request.url}")
        accept = request.headers.get("accept", "")
        print(f"      headers: x-api-key=present (redacted), accept={accept}")
        if request.content:
            print(f"      body:    {clip(request.content.decode(), 420)}")
        response = self._inner.handle_request(request)
        response.read()
        print(f"  ◀── {response.status_code} {response.headers.get('content-type')}")
        body = response.text
        print(f"      {clip(body)}" if body else "      (empty)")
        return httpx.Response(
            response.status_code,
            headers=response.headers,
            content=response.content,
            request=request,
        )


class RecordingWeaveServer:
    """Stands in for wandb.ai and keeps everything that would be uploaded."""

    def __init__(self) -> None:
        self.objs: list[dict[str, Any]] = []
        self.calls: list[tuple[str, Any]] = []
        self.files: list[tuple[str, bytes]] = []
        self.tables: list[list[Any]] = []

    def server_info(self) -> Any:
        from weave.trace_server.service_interface import ServerInfoRes

        return ServerInfoRes(
            min_required_weave_python_version="0.0.0", trace_server_version="wire-trace"
        )

    def ensure_project_exists(self, entity: str, project: str) -> Any:
        from weave.trace_server.service_interface import EnsureProjectExistsRes

        return EnsureProjectExistsRes(project_name=project)

    def obj_create(self, req: Any) -> Any:
        from weave.trace_server import trace_server_interface as tsi

        self.objs.append(dict(req.obj.val))
        return tsi.ObjCreateRes(digest=f"d{len(self.objs)}")

    def call_start(self, req: Any) -> Any:
        from weave.trace_server import trace_server_interface as tsi

        start = req.start
        self.calls.append(("start", {"op": start.op_name, "inputs": start.inputs}))
        return tsi.CallStartRes(id=start.id or "call", trace_id=start.trace_id or "t")

    def call_end(self, req: Any) -> Any:
        from weave.trace_server import trace_server_interface as tsi

        self.calls.append(("end", {"output": req.end.output}))
        return tsi.CallEndRes()

    def file_create(self, req: Any) -> Any:
        # Weave captures an op's source code (enable_code_capture defaults to
        # True) and uploads it alongside the call. Recorded here so it is
        # searched for the API key like everything else.
        from weave.trace_server import trace_server_interface as tsi

        self.files.append((req.name, req.content))
        return tsi.FileCreateRes(digest=f"f{len(self.files)}")

    def table_create(self, req: Any) -> Any:
        from weave.trace_server import trace_server_interface as tsi

        rows = list(getattr(req.table, "rows", []) or [])
        self.tables.append(rows)
        return tsi.TableCreateRes(
            digest=f"t{len(self.tables)}", row_digests=[f"r{i}" for i in range(len(rows))]
        )

    def __getattr__(self, name: str) -> Any:
        # Raise rather than return a stub: weave probes for a batching
        # call_processor with getattr(..., None) and a stub function would be
        # mistaken for one, sending call events down an async path this
        # recorder cannot serve.
        raise AttributeError(name)

    def everything(self) -> str:
        return json.dumps(
            [self.objs, self.calls, [(n, c.decode("utf-8", "replace")) for n, c in self.files], self.tables],
            default=str,
        )


recorder = RecordingWeaveServer()
wc = weave_client.WeaveClient(
    "pair-systems", "goodmem-wire-trace", recorder, ensure_project_exists=False
)
weave_client_context.set_weave_client_global(wc)

http = httpx.Client(
    transport=TracingTransport(),
    base_url=BASE_URL,
    headers={"x-api-key": API_KEY},
    timeout=60.0,
)
sdk = Goodmem(http_client=http)

CHUNKING = {
    "recursive": {
        "chunkSize": 256,
        "chunkOverlap": 25,
        "separators": ["\n\n", "\n", ". ", " ", ""],
        "keepStrategy": "KEEP_END",
        "separatorIsRegex": False,
        "lengthMeasurement": "CHARACTER_COUNT",
    }
}


def make_space(name: str, embedder: str = EMBEDDER) -> str:
    space = sdk.spaces.create(
        name=name,
        space_embedders=[{"embedderId": embedder, "defaultRetrievalWeight": 1.0}],
        default_chunking_config=CHUNKING,
    )
    _spaces.append(space.space_id)
    return space.space_id


# ----------------------------------------------------------------- SETUP
banner("SETUP — a space and three memories (direct SDK calls, to show the shape)")
stamp = int(time.time())
SPACE_NAME = f"wandb-wire-{stamp}"
SPACE = make_space(SPACE_NAME)

FACTS = [
    ("The Weave wire-trace canary is FIG-88-MERIDIAN.", {"category": "canary"}),
    ("Weave records an op call's inputs, output and latency.", {"category": "weave"}),
    ("A GoodMem vector score is a negative inner product.", {"category": "scores"}),
]
memory_ids = []
for text, metadata in FACTS:
    memory = sdk.memories.create(
        space_id=SPACE,
        content_type="text/plain",
        original_content=text,
        metadata=metadata,
    )
    memory_ids.append(memory.memory_id)

print("\n  waiting for indexing (polling suppressed from the trace)…")
deadline = time.time() + 180
import io
import contextlib

while time.time() < deadline:
    with contextlib.redirect_stdout(io.StringIO()):
        states = [sdk.memories.get(id=m).processing_status for m in memory_ids]
    if all(s == "COMPLETED" for s in states):
        break
    time.sleep(2)
print(f"  indexed: {states}")

conn = {"client": sdk}

# ------------------------------------------------------------------- 1
banner("1. SEARCH — one request, and the Weave call it produces")
wc.flush()
before = len(recorder.calls)
retriever = GoodMemRetriever(space_id=SPACE, limit=5, **conn)
out = retriever.search("what is the wire-trace canary")
wc.flush()
print(f"\n  RESULT: {len(out['hits'])} hit(s)  score_kind={out['score_kind']}  partial={out['partial']}  statuses={out['statuses']}")
for hit in out["hits"]:
    print(f"    - [{out['score_kind']} {hit['score']:+.6f}] {hit['chunk_text'].strip()[:62]}")
    print(f"      chunk_id={hit['chunk_id']} memory_id={hit['memory_id']} category={hit['metadata'].get('category')}")

print("\n  UPLOADED TO WANDB.AI for this call:")
for kind, payload in recorder.calls[before:]:
    print(f"    {kind}: {clip(json.dumps(payload, default=str), 640)}")

# ------------------------------------------------------------------- 2
banner("2. RERANKED SEARCH — the server emits FEATURE_DISABLED; it is not a failure")
if RERANKER:
    r = GoodMemRetriever(space_id=SPACE, reranker_id=RERANKER, limit=5, **conn)
    out = r.search("what is the wire-trace canary")
    print(f"\n  RESULT: {len(out['hits'])} hit(s)  score_kind={out['score_kind']}  partial={out['partial']}")
    print(f"  statuses surfaced to the caller: {out['statuses']}  <- FEATURE_DISABLED was classified as informational")
    for hit in out["hits"]:
        print(f"    - [reranker {hit['score']:+.6f}] {hit['chunk_text'].strip()[:62]}")
else:
    print("\n  (skipped: GOODMEM_RERANKER_ID not set)")

# ------------------------------------------------------------------- 3
banner("3. BROKEN RERANKER — real statuses, surfaced and marked partial")
r = GoodMemRetriever(
    space_id=SPACE, reranker_id="00000000-0000-0000-0000-000000000000", **conn
)
out = r.search("canary")
print(f"\n  RESULT: {len(out['hits'])} hit(s)  partial={out['partial']}")
for status in out["statuses"]:
    print(f"    STATUS {status.get('code')}: {str(status.get('message'))[:96]}")

# ------------------------------------------------------------------- 4
banner("4. EMPTY SPACE — one request, returns immediately (0.1.0 polled for 10s)")
EMPTY = make_space(f"wandb-wire-empty-{stamp}")
started = time.time()
out = GoodMemRetriever(space_id=EMPTY, **conn).search("anything at all")
print(f"\n  RESULT: {len(out['hits'])} hit(s)  partial={out['partial']}  in {time.time() - started:.2f}s")

# ------------------------------------------------------------------- 5
banner("5. METADATA FILTER — escaped server-side expression in the request body")
out = GoodMemRetriever(space_id=SPACE, **conn).search(
    "scores", metadata_filter={"category": "scores"}
)
print(f"\n  RESULT (matching filter): {len(out['hits'])} hit(s)")
out = GoodMemRetriever(space_id=SPACE, **conn).search(
    "canary", metadata_filter={"category": "o'brien \\ audit"}
)
print(f"  RESULT (apostrophe + backslash value): {len(out['hits'])} hit(s) — escaped, no 400")

# ------------------------------------------------------------------- 6
banner("6. ATTACH BY NAME — reuse when the embedder matches, refuse when it does not")
r = GoodMemRetriever(space_name=SPACE_NAME, embedder_id=EMBEDDER, **conn)
out = r.search("canary")
print(f"\n  RESULT: resolved to {out['space_ids']}  (expected [{SPACE}])")

print("\n  …now the same name with a different embedder:")
r = GoodMemRetriever(
    space_name=SPACE_NAME, embedder_id="00000000-0000-0000-0000-000000000000", **conn
)
try:
    r.search("canary")
    print("  NO ERROR — this would be the 0.1.0 bug")
except GoodMemSpaceError as exc:
    print(f"\n  REFUSED: {exc}")
    print("  (0.1.0 returned reused:true and echoed back the embedder you asked for)")

# ------------------------------------------------------------------- 7
banner("7. THE W&B WIRE — publish the model, and check the upload for the API key")
wc.flush()
before_objs = len(recorder.objs)
model = GoodMemRetrievalModel(
    space_id=SPACE,
    limit=5,
    name="goodmem-baseline",
    base_url=BASE_URL,
    api_key=API_KEY,
)
weave.publish(model)
wc.flush()
print("\n  PAYLOAD UPLOADED TO WANDB.AI (this is the whole object record):")
for obj in recorder.objs[before_objs:]:
    print(f"    {json.dumps(obj, default=str)}")

leaked = SECRET_MARKER in recorder.everything()
print(f"\n  API key present anywhere in everything uploaded so far? {'*** YES — LEAK ***' if leaked else 'no'}")
print(f"  model.model_dump() = {json.dumps(model.model_dump(), default=str)}")

print("\n  For contrast — the naive design, a weave.Model with an api_key field:")


class NaiveModel(weave.Model):
    base_url: str = "https://localhost:8080"
    api_key: str = "gm_EXAMPLE_NOT_A_REAL_KEY_00000"

    @weave.op
    def predict(self, question: str) -> str:
        return question


wc.flush()
before_objs = len(recorder.objs)
weave.publish(NaiveModel(name="naive"))
wc.flush()
for obj in recorder.objs[before_objs:]:
    print(f"    {json.dumps(obj, default=str)}")
print("    ^ weave's should_redact() does not run on this path — the field goes up verbatim")

# ------------------------------------------------------------------- 8
banner("8. EVALUATION — scorers over a live retrieval")
model = GoodMemRetrievalModel(space_id=SPACE, limit=5, **conn)
out = model.predict("what is the wire-trace canary")
canary_id = memory_ids[0]
print(f"\n  predict() -> {len(out['hits'])} hit(s), top memory_id={out['hits'][0]['memory_id'] if out['hits'] else None}")
print(f"  ground truth canary memory_id={canary_id}")
print("\n  SCORER RESULTS:")
print(f"    RecallAtK(k=5)   -> {RecallAtK(k=5).score(output=out, expected_memory_ids=[canary_id])}")
print(f"    MRR()            -> {MRR().score(output=out, expected_memory_ids=[canary_id])}")
print(f"    FactRecall()     -> {FactRecall().score(output=out, expected_text='FIG-88-MERIDIAN')}")
print(f"    RetrievalHealth()-> {RetrievalHealth().score(output=out)}")

# ------------------------------------------------------------------- 9
banner("9. trace_chunk_text=False — ids and scores traced, corpus text withheld")
wc.flush()
before = len(recorder.calls)
r = GoodMemRetriever(space_id=SPACE, trace_chunk_text=False, **conn)
out = r.search("what is the wire-trace canary")
wc.flush()
print(f"\n  RESULT: {len(out['hits'])} hit(s); chunk_text = {[h['chunk_text'] for h in out['hits']]}")
print(f"  scores still present: {[round(h['score'], 4) for h in out['hits']]}")
uploaded = json.dumps(recorder.calls[before:], default=str)
print(f"  events uploaded for this call: {[kind for kind, _ in recorder.calls[before:]]}")
print(f"  'FIG-88-MERIDIAN' present in what was uploaded for this call? {'*** YES ***' if 'FIG-88-MERIDIAN' in uploaded else 'no'}")
print(f"  what the trace carries instead: {json.dumps([h for h in out['hits']][:1], default=str)[:220]}")

# --------------------------------------------------------------- TEARDOWN
banner("TEARDOWN — delete every space created, then verify")
for space_id in _spaces:
    sdk.spaces.delete(id=space_id)

leftovers = [
    s.name
    for s in sdk.spaces.list(name_filter=f"wandb-wire-{stamp}", max_items=50)
]
wc.flush()
print(f"\n  deleted {len(_spaces)} spaces | leaked: {leftovers or 'none'}")
print(f"  total GoodMem HTTP requests traced: {_n}")
print(
    f"  total payloads that would have reached wandb.ai: {len(recorder.objs)} objects, "
    f"{len(recorder.calls)} call events, {len(recorder.files)} source files"
)
print(f"  API key anywhere in those payloads: {'*** YES ***' if SECRET_MARKER in recorder.everything() else 'no'}")
