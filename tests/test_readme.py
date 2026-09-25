"""The README's Python snippets run as written.

Only the outside world is swapped, never the snippet text:

* ``goodmem_wandb._connection.Goodmem`` builds the real SDK on an httpx
  MockTransport that replays the captured fixtures, keyed on the
  ``GOODMEM_BASE_URL`` / ``GOODMEM_API_KEY`` the snippets read from the
  environment;
* ``weave.init`` installs an in-memory trace server instead of logging in to
  W&B;
* the id placeholders (``"…"``, ``"01a0…"``) become ids from the fixtures.

The blocks run in order in one namespace, as a reader would paste them.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import re
from typing import Any

from goodmem import Goodmem
import httpx
import pytest

from tests.conftest import InMemoryTraceServer, load_bytes, load_json

README = Path(__file__).resolve().parent.parent / "README.md"
BASE_URL = "https://goodmem.readme.test"
API_KEY = "gm_readme_test_key_not_real"

MEMORY = "01a0ce67-ff50-764a-bbe0-a098fe9c250f"
RERANKER = "019cfda4-7e2f-743c-9edb-e469a97b95c6"
EMBEDDER = "019cfd1c-c033-7517-b7de-f73941a0464b"
PLACEHOLDERS = [
    ('"01a0…"', f'"{MEMORY}"'),
    ('reranker_id="…"', f'reranker_id="{RERANKER}"'),
    ('embedder_id="…"', f'embedder_id="{EMBEDDER}"'),
]


def python_blocks() -> list[str]:
    text = README.read_text(encoding="utf-8")
    return [
        body
        for lang, body in re.findall(r"```(\w*)\n(.*?)```", text, re.DOTALL)
        if lang == "python"
    ]


class TracingServer(InMemoryTraceServer):
    """The recorder, with enough of the call path for weave.Evaluation."""

    # Without these the catch-all stub poses as a batch processor and a limit.
    call_processor = None
    remote_request_bytes_limit = 32 * 1024 * 1024

    def call_start(self, req: Any) -> Any:
        from weave.trace_server import trace_server_interface as tsi

        self.calls.append(("call_start", (req,)))
        return tsi.CallStartRes(id=req.start.id, trace_id=req.start.trace_id)

    def feedback_create(self, req: Any) -> Any:
        import datetime

        from weave.trace_server import trace_server_interface as tsi

        self.calls.append(("feedback_create", (req,)))
        return tsi.FeedbackCreateRes(
            id=f"f{len(self.calls)}",
            created_at=datetime.datetime.now(datetime.timezone.utc),
            wb_user_id="readme",
            payload={},
        )

    def table_create(self, req: Any) -> Any:
        from weave.trace_server import trace_server_interface as tsi

        rows = req.table.rows
        return tsi.TableCreateRes(
            digest=f"t{len(self.calls)}",
            row_digests=[f"r{i}" for i in range(len(rows))],
        )

    def started(self, op_fragment: str) -> int:
        return sum(
            1
            for name, args in self.calls
            if name == "call_start" and op_fragment in str(args)
        )


class GoodMemStandIn:
    """Serve the captured bytes for whatever the snippets ask for."""

    def __init__(self) -> None:
        docs = copy.deepcopy(load_json("spaces_list.json")["spaces"][0])
        docs["name"] = "docs"
        self.spaces = [docs]
        self.retrieves: list[dict[str, Any]] = []
        self.keys_seen: set[str | None] = set()

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.keys_seen.add(request.headers.get("x-api-key"))
        path = request.url.path
        if request.method == "GET" and path == "/v1/spaces":
            name = request.url.params.get("name_filter", "")
            return httpx.Response(
                200, json={"spaces": [s for s in self.spaces if name in s["name"]]}
            )
        if request.method == "POST" and path == "/v1/memories:retrieve":
            body = json.loads(request.content)
            self.retrieves.append(body)
            if body.get("postProcessor"):
                fixture = "retrieve_reranked.ndjson"
            elif any(k.get("filter") for k in body.get("spaceKeys", [])):
                fixture = "retrieve_filtered.ndjson"
            else:
                fixture = "retrieve_vector.ndjson"
            return httpx.Response(
                200,
                content=load_bytes(fixture),
                headers={"Content-Type": "application/x-ndjson"},
            )
        raise AssertionError(f"unrouted {request.method} {path}")

    def factory(self, *, base_url: str, api_key: str, **_: Any) -> Goodmem:
        # A fresh client per operation: the connection closes what it opens.
        return Goodmem(
            http_client=httpx.Client(
                transport=httpx.MockTransport(self.handler),
                base_url=base_url,
                headers={"x-api-key": api_key},
            )
        )


@pytest.fixture
def readme_env(monkeypatch: pytest.MonkeyPatch) -> Any:
    import weave
    from weave.trace import weave_client
    from weave.trace.context import weave_client_context

    stand_in = GoodMemStandIn()
    trace = TracingServer()

    def init(project: str, *args: Any, **kwargs: Any) -> Any:
        wc = weave_client.WeaveClient(
            "readme", project, trace, ensure_project_exists=False
        )
        weave_client_context.set_weave_client_global(wc)
        return wc

    monkeypatch.setenv("GOODMEM_BASE_URL", BASE_URL)
    monkeypatch.setenv("GOODMEM_API_KEY", API_KEY)
    monkeypatch.setattr("goodmem_wandb._connection.Goodmem", stand_in.factory)
    monkeypatch.setattr(weave, "init", init)
    yield stand_in, trace
    weave_client_context.set_weave_client_global(None)


def test_every_readme_python_block_runs(readme_env: Any) -> None:
    stand_in, _trace = readme_env
    blocks = python_blocks()
    assert len(blocks) == 6, "README python blocks changed; revisit this test"

    namespace: dict[str, Any] = {"__name__": "__readme__"}
    for index, block in enumerate(blocks):
        for placeholder, value in PLACEHOLDERS:
            block = block.replace(placeholder, value)
        # Executing the README is the point of this test.
        exec(compile(block, f"README.md python block {index}", "exec"), namespace)  # noqa: S102

    assert stand_in.keys_seen == {API_KEY}, (
        "credentials should come from the environment"
    )
    # Trace a retrieval: one search. Filtering: one search, both filters AND-ed.
    assert stand_in.retrieves[0]["message"] == "how do I rotate an API key?"
    filtered = [b for b in stand_in.retrieves if b["message"] == "refunds"]
    assert len(filtered) == 1
    expression = filtered[0]["spaceKeys"][0]["filter"]
    assert "'billing'" in expression and "'en'" in expression and " AND " in expression


def test_the_readme_evaluation_actually_runs_both_models(readme_env: Any) -> None:
    """``Evaluation.evaluate`` is a coroutine. Called bare, as the README once
    showed, it returns an un-awaited coroutine: no model is asked anything, no
    scorer runs and nothing reaches the Weave UI."""
    stand_in, trace = readme_env
    evaluate_block = next(b for b in python_blocks() if "weave.Evaluation(" in b)
    for placeholder, value in PLACEHOLDERS:
        evaluate_block = evaluate_block.replace(placeholder, value)

    exec(compile(evaluate_block, "README.md evaluation block", "exec"), {})  # noqa: S102

    assert trace.started("Evaluation.evaluate") == 2
    assert trace.started("GoodMemRetrievalModel.predict") == 2
    # One retrieval per model, the second through the reranker.
    assert len(stand_in.retrieves) == 2
    assert "postProcessor" not in stand_in.retrieves[0]
    assert RERANKER in json.dumps(stand_in.retrieves[1])


def test_the_demo_script_actually_runs_its_evaluations(
    readme_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import runpy

    demo = README.parent / "examples" / "demo.py"
    if not demo.exists():  # the sdist ships tests but not examples
        pytest.skip("examples/demo.py is not in this tree")
    stand_in, trace = readme_env
    monkeypatch.setenv("GOODMEM_SPACE_ID", "01a0ce67-ff3e-748d-aba9-0e36a8da091c")
    monkeypatch.setenv("GOODMEM_RERANKER_ID", RERANKER)

    runpy.run_path(str(demo), run_name="__main__")

    assert trace.started("Evaluation.evaluate") == 2
    assert trace.started("GoodMemRetrievalModel.predict") == 2
    # The traced search, then one retrieval per evaluated model.
    assert len(stand_in.retrieves) == 3


def test_the_model_docstring_example_actually_runs_its_evaluation(
    readme_env: Any,
) -> None:
    import textwrap

    from goodmem_wandb import GoodMemRetrievalModel

    doc = GoodMemRetrievalModel.__doc__ or ""
    example = textwrap.dedent(doc.split("::\n", 1)[1].split("\n\n    Changing")[0])
    for placeholder, value in PLACEHOLDERS:
        example = example.replace(placeholder, value)
    _stand_in, trace = readme_env

    exec(compile(example, "GoodMemRetrievalModel docstring", "exec"), {})  # noqa: S102

    assert trace.started("Evaluation.evaluate") == 1
    assert trace.started("GoodMemRetrievalModel.predict") == 1
