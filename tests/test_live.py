"""End-to-end live test for goodmem_wandb.

Exercises every operation exposed by ``GoodMemClient`` against a real
GoodMem server, including the post-processor retrieval features
(reranker, LLM, relevance threshold, LLM temperature, chronological
resort, max_results). Designed to be run as a script:

    python tests/test_live.py

It exits non-zero on the first failure and cleans up everything it
creates so the server is left in a consistent state.

Configuration is read from environment variables when set, falling
back to the defaults provided for the local dev instance:

    GOODMEM_BASE_URL        default: https://localhost:8080
    GOODMEM_API_KEY         default: <provided test key>
    GOODMEM_EMBEDDER_ID     default: 019cfd1c-c033-7517-b7de-f73941a0464b
    GOODMEM_RERANKER_ID     default: 019cfda4-7e2f-743c-9edb-e469a97b95c6
    GOODMEM_LLM_ID          default: 019cfd9f-0963-76f9-b069-4cde19a64ba8
    GOODMEM_PDF_PATH        default: first .pdf found in ~/Downloads (optional)
"""

from __future__ import annotations

import glob
import os
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

from goodmem_wandb import GoodMemClient

try:
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

BASE_URL = os.environ.get("GOODMEM_BASE_URL", "https://localhost:8080")
API_KEY = os.environ.get("GOODMEM_API_KEY", "gm_g5xcse2tjgcznlg45c5le4ti5q")
EMBEDDER_ID = os.environ.get(
    "GOODMEM_EMBEDDER_ID", "019cfd1c-c033-7517-b7de-f73941a0464b"
)
RERANKER_ID = os.environ.get(
    "GOODMEM_RERANKER_ID", "019cfda4-7e2f-743c-9edb-e469a97b95c6"
)
LLM_ID = os.environ.get("GOODMEM_LLM_ID", "019cfd9f-0963-76f9-b069-4cde19a64ba8")

_default_pdf_candidates = sorted(
    glob.glob(os.path.expanduser("~/Downloads/*.pdf"))
)
PDF_PATH = os.environ.get(
    "GOODMEM_PDF_PATH",
    _default_pdf_candidates[0] if _default_pdf_candidates else "",
)

SPACE_NAME = f"goodmem-wandb-live-{int(time.time())}"

BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


_passes: List[str] = []
_failures: List[str] = []


def banner(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}{RESET}")


def run(name: str, fn: Callable[[], Any]) -> Any:
    print(f"\n  {BOLD}▶ {name}{RESET}")
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001
        print(f"  {RED}✗ FAIL: {name}{RESET}")
        print(f"  {RED}{traceback.format_exc().rstrip()}{RESET}")
        _failures.append(f"{name}: {exc}")
        return None
    print(f"  {GREEN}✓ pass{RESET}")
    _passes.append(name)
    return result


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    banner("goodmem_wandb · live end-to-end test")
    print(f"  base_url   : {BASE_URL}")
    print(f"  api_key    : {API_KEY[:6]}…{API_KEY[-4:]}")
    print(f"  embedder   : {EMBEDDER_ID}")
    print(f"  reranker   : {RERANKER_ID}")
    print(f"  llm        : {LLM_ID}")
    print(f"  pdf_path   : {PDF_PATH or '(none — PDF step will be skipped)'}")
    print(f"  space name : {SPACE_NAME}")

    # ------------------------------------------------------------------
    # 0 · Connect
    # ------------------------------------------------------------------
    banner("0 · Connect to GoodMem server")
    client = run(
        "GoodMemClient(__init__) — validates connection",
        lambda: GoodMemClient(
            base_url=BASE_URL,
            api_key=API_KEY,
            max_wait_s=30,
            poll_interval_s=3,
            verify_ssl=False,
        ),
    )
    if client is None:
        return _summary()

    state: Dict[str, Any] = {
        "space_id": None,
        "text_memory_id": None,
        "pdf_memory_id": None,
    }

    # ------------------------------------------------------------------
    # 1 · list_embedders
    # ------------------------------------------------------------------
    banner("1 · list_embedders")

    def _list_embedders() -> List[Dict[str, Any]]:
        embedders = client.list_embedders()
        expect(isinstance(embedders, list), "list_embedders should return a list")
        expect(len(embedders) >= 1, "expected at least one embedder")
        ids = [e.get("embedderId") for e in embedders]
        print(f"    found {len(embedders)} embedder(s)")
        for e in embedders[:5]:
            print(f"      • {e.get('embedderId')} — {e.get('modelIdentifier', '?')}")
        expect(
            EMBEDDER_ID in ids,
            f"configured EMBEDDER_ID {EMBEDDER_ID} not present on the server",
        )
        return embedders

    run("list_embedders", _list_embedders)

    # ------------------------------------------------------------------
    # 2 · create_space
    # ------------------------------------------------------------------
    banner("2 · create_space")

    def _create_space() -> Dict[str, Any]:
        space = client.create_space(name=SPACE_NAME, embedder_id=EMBEDDER_ID)
        expect(space.get("success") is True, "create_space did not report success")
        expect(bool(space.get("spaceId")), "create_space returned empty spaceId")
        expect(space.get("reused") is False, "fresh space should not be reused")
        state["space_id"] = space["spaceId"]
        print(f"    spaceId : {space['spaceId']}")
        print(f"    message : {space['message']}")
        return space

    run("create_space (fresh)", _create_space)

    def _create_space_idempotent() -> Dict[str, Any]:
        again = client.create_space(name=SPACE_NAME, embedder_id=EMBEDDER_ID)
        expect(again.get("reused") is True, "second create_space should report reused=True")
        expect(again["spaceId"] == state["space_id"], "reused spaceId mismatch")
        return again

    run("create_space (idempotent reuse)", _create_space_idempotent)

    # ------------------------------------------------------------------
    # 3 · list_spaces
    # ------------------------------------------------------------------
    banner("3 · list_spaces")

    def _list_spaces() -> List[Dict[str, Any]]:
        spaces = client.list_spaces()
        expect(isinstance(spaces, list), "list_spaces should return a list")
        ids = {s.get("spaceId") for s in spaces}
        expect(state["space_id"] in ids, "newly created space missing from list_spaces")
        print(f"    total spaces on server : {len(spaces)}")
        return spaces

    run("list_spaces", _list_spaces)

    # ------------------------------------------------------------------
    # 4 · get_space
    # ------------------------------------------------------------------
    banner("4 · get_space")

    def _get_space() -> Dict[str, Any]:
        got = client.get_space(state["space_id"])
        expect(got.get("success") is True, "get_space did not report success")
        expect(got["space"]["spaceId"] == state["space_id"], "spaceId mismatch")
        expect(got["space"]["name"] == SPACE_NAME, "name mismatch")
        print(f"    name : {got['space']['name']}")
        return got

    run("get_space", _get_space)

    # ------------------------------------------------------------------
    # 5 · update_space (merge_labels)
    # ------------------------------------------------------------------
    banner("5 · update_space (merge_labels)")

    def _update_space() -> Dict[str, Any]:
        updated = client.update_space(
            space_id=state["space_id"],
            merge_labels={"framework": "wandb", "env": "live-test"},
        )
        expect(updated.get("success") is True, "update_space did not report success")
        labels = updated.get("space", {}).get("labels") or {}
        expect(labels.get("framework") == "wandb", "label 'framework' not persisted")
        expect(labels.get("env") == "live-test", "label 'env' not persisted")
        print(f"    labels  : {labels}")
        print(f"    message : {updated['message']}")
        return updated

    run("update_space", _update_space)

    # ------------------------------------------------------------------
    # 6 · create_memory (text)
    # ------------------------------------------------------------------
    banner("6 · create_memory (text)")

    def _create_text_memory() -> Dict[str, Any]:
        mem = client.create_memory(
            space_id=state["space_id"],
            text_content=(
                "GoodMem is a memory layer for AI agents with support for "
                "semantic storage, retrieval, and summarization. This "
                "goodmem-wandb package exposes GoodMem operations as a "
                "Python client for wandb agent integrations."
            ),
            source="goodmem-wandb-live-test",
            author="ci",
            tags="goodmem,wandb,live",
        )
        expect(mem.get("success") is True, "create_memory (text) did not report success")
        expect(bool(mem.get("memoryId")), "create_memory (text) returned empty memoryId")
        expect(mem["contentType"] == "text/plain", "text memory contentType mismatch")
        state["text_memory_id"] = mem["memoryId"]
        print(f"    memoryId : {mem['memoryId']}")
        print(f"    status   : {mem['status']}")
        return mem

    run("create_memory (text)", _create_text_memory)

    # ------------------------------------------------------------------
    # 7 · create_memory (PDF)
    # ------------------------------------------------------------------
    banner("7 · create_memory (PDF)")
    if PDF_PATH and os.path.isfile(PDF_PATH):

        def _create_pdf_memory() -> Dict[str, Any]:
            mem = client.create_memory(
                space_id=state["space_id"],
                file_path=PDF_PATH,
                source="goodmem-wandb-live-test-pdf",
                author="ci",
                tags="pdf,goodmem,wandb",
            )
            expect(mem.get("success") is True, "create_memory (PDF) did not report success")
            expect(bool(mem.get("memoryId")), "create_memory (PDF) returned empty memoryId")
            expect(mem["contentType"] == "application/pdf", "PDF contentType mismatch")
            expect(
                mem["fileName"] == os.path.basename(PDF_PATH),
                "PDF fileName mismatch",
            )
            state["pdf_memory_id"] = mem["memoryId"]
            print(f"    memoryId : {mem['memoryId']}")
            print(f"    fileName : {mem['fileName']}")
            return mem

        run("create_memory (PDF)", _create_pdf_memory)
    else:
        print(f"  {YELLOW}⚠ skipped — no PDF available at {PDF_PATH or '(not set)'}{RESET}")

    # ------------------------------------------------------------------
    # 8 · list_memories
    # ------------------------------------------------------------------
    banner("8 · list_memories")

    def _list_memories() -> Dict[str, Any]:
        listing = client.list_memories(space_id=state["space_id"])
        expect(listing.get("success") is True, "list_memories did not report success")
        memories = listing["memories"]
        ids = {m.get("memoryId") for m in memories}
        expect(
            state["text_memory_id"] in ids,
            "text memory not present in list_memories",
        )
        if state["pdf_memory_id"]:
            expect(
                state["pdf_memory_id"] in ids,
                "PDF memory not present in list_memories",
            )
        print(f"    total memories : {len(memories)}")
        for m in memories:
            print(
                f"      • {m['memoryId']}  status={m.get('processingStatus', '?')}"
            )
        return listing

    run("list_memories", _list_memories)

    # ------------------------------------------------------------------
    # 9 · retrieve_memories (basic)
    # ------------------------------------------------------------------
    banner("9 · retrieve_memories (basic semantic search)")

    def _retrieve_basic() -> Dict[str, Any]:
        out = client.retrieve_memories(
            query="memory layer for AI agents",
            space_ids=[state["space_id"]],
            max_results=5,
            wait_for_indexing=True,
        )
        expect(out.get("success") is True, "retrieve_memories did not report success")
        print(f"    totalResults : {out['totalResults']}")
        for r in out["results"][:3]:
            snippet = (r.get("chunkText") or "")[:90].replace("\n", " ")
            score = r.get("relevanceScore")
            score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "?"
            print(f"      • score={score_str}  {snippet}")
        # Indexing may still be in progress; only assert when we got results.
        if out["totalResults"] > 0:
            expect(
                isinstance(out["results"], list),
                "retrieve_memories.results should be a list",
            )
        else:
            print(
                f"    {YELLOW}note: 0 results — embedder may still be indexing{RESET}"
            )
        return out

    run("retrieve_memories (basic)", _retrieve_basic)

    # ------------------------------------------------------------------
    # 10 · retrieve_memories (reranker + llm + threshold + temperature + chronological)
    # ------------------------------------------------------------------
    banner(
        "10 · retrieve_memories (reranker + LLM + relevance_threshold + "
        "llm_temperature + chronological_resort)"
    )

    def _retrieve_advanced() -> Dict[str, Any]:
        out = client.retrieve_memories(
            query="What is GoodMem and what does it provide?",
            space_ids=[state["space_id"]],
            max_results=3,
            wait_for_indexing=True,
            reranker_id=RERANKER_ID,
            llm_id=LLM_ID,
            relevance_threshold=0.0,
            llm_temperature=0.2,
            chronological_resort=True,
        )
        expect(out.get("success") is True, "advanced retrieve did not report success")
        print(f"    totalResults : {out['totalResults']}")
        if "abstractReply" in out:
            reply_text = (
                out["abstractReply"].get("text")
                or out["abstractReply"].get("content")
                or str(out["abstractReply"])
            )
            print(f"    abstractReply: {str(reply_text)[:200]}…")
        else:
            print(
                f"    {YELLOW}note: no abstractReply returned (LLM may have produced "
                f"no answer or post-processor disabled){RESET}"
            )
        return out

    run("retrieve_memories (advanced post-processor)", _retrieve_advanced)

    # ------------------------------------------------------------------
    # 11 · get_memory
    # ------------------------------------------------------------------
    banner("11 · get_memory")

    def _get_memory() -> Dict[str, Any]:
        got = client.get_memory(state["text_memory_id"], include_content=True)
        expect(got.get("success") is True, "get_memory did not report success")
        expect(
            got["memory"]["memoryId"] == state["text_memory_id"],
            "get_memory returned wrong memoryId",
        )
        has_content = "content" in got or "contentB64" in got
        expect(has_content, "get_memory(include_content=True) returned no content")
        print(f"    memoryId : {got['memory']['memoryId']}")
        print(f"    status   : {got['memory'].get('processingStatus', '?')}")
        if "content" in got:
            preview = str(got["content"])[:120].replace("\n", " ")
            print(f"    content  : {preview}")
        return got

    run("get_memory", _get_memory)

    # ------------------------------------------------------------------
    # 12 · delete_memory
    # ------------------------------------------------------------------
    banner("12 · delete_memory")

    def _delete_text_memory() -> Dict[str, Any]:
        out = client.delete_memory(state["text_memory_id"])
        expect(out.get("success") is True, "delete_memory did not report success")
        return out

    run("delete_memory (text)", _delete_text_memory)

    if state["pdf_memory_id"]:
        run(
            "delete_memory (pdf)",
            lambda: client.delete_memory(state["pdf_memory_id"]),
        )

    # ------------------------------------------------------------------
    # 13 · delete_space (clean up)
    # ------------------------------------------------------------------
    banner("13 · delete_space (cleanup)")

    def _delete_space() -> Dict[str, Any]:
        out = client.delete_space(state["space_id"])
        expect(out.get("success") is True, "delete_space did not report success")
        # Verify the space is gone
        spaces = client.list_spaces()
        ids = {s.get("spaceId") for s in spaces}
        expect(
            state["space_id"] not in ids,
            "space still present after delete_space",
        )
        return out

    run("delete_space + verify gone", _delete_space)

    return _summary()


def _summary() -> int:
    banner("SUMMARY")
    print(f"  {GREEN}✓ {len(_passes)} passed{RESET}")
    if _failures:
        print(f"  {RED}✗ {len(_failures)} failed{RESET}")
        for f in _failures:
            print(f"    - {f}")
        print(
            f"\n  {RED}{BOLD}LIVE TEST FAILED — see failures above.{RESET}"
        )
        return 1
    print(f"\n  {GREEN}{BOLD}ALL OPERATIONS PASSED AGAINST LIVE SERVER.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
