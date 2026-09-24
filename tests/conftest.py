"""Test fixtures.

Two recorders, no network:

* ``client`` is a real ``goodmem.Goodmem`` driven by an httpx MockTransport
  that replays bytes captured from a live server (v1.0.320) into the real SDK
  decoders. Nothing about the wire format is invented here.
* ``weave_recorder`` is a real ``weave.trace.weave_client.WeaveClient`` whose
  trace server is in-memory, so a test can assert on exactly what would have
  been uploaded to W&B.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from goodmem import Goodmem
import httpx
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def load_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def load_json(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text())


def ndjson_events(name: str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in load_bytes(name).decode().strip().split("\n")
        if line.strip()
    ]


def to_ndjson(events: list[dict[str, Any]]) -> bytes:
    return ("\n".join(json.dumps(e) for e in events) + "\n").encode()


class Recorder:
    """Route requests to canned responses and remember what was sent."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.bodies: list[Any] = []
        self._routes: list[tuple[str, str, Any]] = []

    def route(self, method: str, path: str, response: httpx.Response) -> None:
        self._routes.append((method.upper(), path, response))

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        try:
            self.bodies.append(json.loads(request.content) if request.content else None)
        except ValueError:
            self.bodies.append(request.content)
        for method, path, response in self._routes:
            if request.method == method and request.url.path == path:
                return response
        raise AssertionError(f"unrouted {request.method} {request.url.path}")

    @property
    def last_body(self) -> Any:
        return self.bodies[-1]

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


def ndjson_response(events: list[dict[str, Any]]) -> httpx.Response:
    return httpx.Response(
        200,
        content=to_ndjson(events),
        headers={"Content-Type": "application/x-ndjson; charset=utf-8"},
    )


@pytest.fixture
def recorder() -> Recorder:
    return Recorder()


@pytest.fixture
def client(recorder: Recorder) -> Goodmem:
    """A real SDK client whose transport replays fixtures.

    With an injected http_client the SDK does not apply base_url or the API
    key itself, so both live on the httpx client.
    """
    http_client = httpx.Client(
        transport=recorder.transport(),
        base_url="https://goodmem.test",
        headers={"x-api-key": "gm_test_key_not_a_real_credential"},
    )
    # The SDK refuses base_url/api_key alongside an injected client: both
    # belong on the httpx.Client above.
    return Goodmem(http_client=http_client)


# ------------------------------------------------------------------- weave
class InMemoryTraceServer:
    """Enough of the Weave trace server to see what a publish would upload."""

    def __init__(self) -> None:
        self.objs: list[dict[str, Any]] = []
        self.calls: list[tuple[str, Any]] = []

    def server_info(self) -> Any:
        from weave.trace_server.service_interface import ServerInfoRes

        return ServerInfoRes(
            min_required_weave_python_version="0.0.0", trace_server_version="test"
        )

    def ensure_project_exists(self, entity: str, project: str) -> Any:
        from weave.trace_server.service_interface import EnsureProjectExistsRes

        return EnsureProjectExistsRes(project_name=project)

    def obj_create(self, req: Any) -> Any:
        from weave.trace_server import trace_server_interface as tsi

        self.objs.append(dict(req.obj.val))
        return tsi.ObjCreateRes(digest=f"d{len(self.objs)}")

    def __getattr__(self, name: str) -> Any:
        def stub(*args: Any, **kwargs: Any) -> None:
            self.calls.append((name, args))

        return stub

    def uploaded_text(self) -> str:
        return json.dumps([self.objs, [str(c) for c in self.calls]], default=str)


@pytest.fixture
def weave_recorder(monkeypatch: pytest.MonkeyPatch) -> InMemoryTraceServer:
    from weave.trace import weave_client
    from weave.trace.context import weave_client_context

    server = InMemoryTraceServer()
    wc = weave_client.WeaveClient(
        "test-entity", "test-project", server, ensure_project_exists=False
    )
    weave_client_context.set_weave_client_global(wc)
    yield server
    weave_client_context.set_weave_client_global(None)
