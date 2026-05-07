"""Tests for CLI tool retrieval indexing and dispatch."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.prebuilt.tool_node import ToolCallRequest

from deepagents_cli.main import parse_args
from deepagents_cli.tool_retrieval import (
    ToolRetrievalError,
    ToolRetrievalIndex,
    ToolRetrievalMiddleware,
    ToolRetrievalResult,
    _ThreadRetrievalState,
    _tool_records,
    load_or_build_index,
    select_tools_for_query,
)


class _FakeIndexFlatIP:
    def __init__(self, dimensions: int) -> None:
        self.d = dimensions
        self.vectors = np.empty((0, dimensions), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return int(self.vectors.shape[0])

    def add(self, matrix: np.ndarray) -> None:
        self.vectors = np.asarray(matrix, dtype=np.float32)

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = self.vectors @ query[0]
        order = np.argsort(-scores)[:k]
        return scores[order].reshape(1, -1), order.astype(np.int64).reshape(1, -1)


class _FakeFaiss:
    IndexFlatIP = _FakeIndexFlatIP

    @staticmethod
    def write_index(index: _FakeIndexFlatIP, path: str) -> None:
        payload = {
            "d": index.d,
            "vectors": index.vectors.tolist(),
        }
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    @staticmethod
    def read_index(path: str) -> _FakeIndexFlatIP:
        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        index = _FakeIndexFlatIP(int(payload["d"]))
        index.add(np.asarray(payload["vectors"], dtype=np.float32))
        return index


def _tool_schema(name: str, description: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _embedding(texts: list[str], _config: dict[str, object]) -> np.ndarray:
    vectors = []
    for text in texts:
        lowered = text.lower()
        if "read" in lowered:
            vectors.append([1.0, 0.0])
        elif "write" in lowered:
            vectors.append([0.0, 1.0])
        else:
            vectors.append([0.5, 0.5])
    return np.asarray(vectors, dtype=np.float32)


def test_index_builds_selects_and_reuses_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Selection should use the cached FAISS index for the same schema hash."""
    monkeypatch.setitem(sys.modules, "faiss", _FakeFaiss)
    config = {
        "top_k": 1,
        "cache_dir": str(tmp_path),
        "model_cache_dir": str(tmp_path / "models"),
        "model": "local/test-model",
    }
    tools = [
        _tool_schema("read_file", "Read project files"),
        _tool_schema("write_file", "Write project files"),
    ]

    prepared = load_or_build_index(tools, config, platform="cli", embedder=_embedding)
    first = select_tools_for_query(
        tools,
        "read project files",
        config,
        platform="cli",
        embedder=_embedding,
        prepared_index=prepared,
    )
    second = load_or_build_index(tools, config, platform="cli", embedder=_embedding)

    assert first.selected_names == ["read_file"]
    assert second.metadata["schema_hash"] == prepared.metadata["schema_hash"]
    assert second.index_path.exists()
    assert second.metadata_path.exists()


def test_select_tools_for_query_rejects_empty_query() -> None:
    with pytest.raises(ToolRetrievalError, match="query is empty"):
        select_tools_for_query([_tool_schema("read_file", "Read files")], " ", {})


def test_prepare_request_hides_native_tools_and_adds_guidance() -> None:
    def read_file(path: str) -> str:
        """Read a file."""
        return path

    def write_file(path: str, content: str) -> str:
        """Write a file."""
        return f"{path}:{content}"

    native_tool = StructuredTool.from_function(
        read_file,
        name="read_file",
        description="Read a file",
    )
    second_tool = StructuredTool.from_function(
        write_file,
        name="write_file",
        description="Write a file",
    )

    def load_index(*_args: object, **_kwargs: object) -> ToolRetrievalIndex:
        return ToolRetrievalIndex(
            index=object(),
            metadata={},
            entries=[{"name": "read_file"}],
            vector_dimensions=2,
            index_path=Path("/tmp/index.faiss"),
            metadata_path=Path("/tmp/index.meta.json"),
        )

    middleware = ToolRetrievalMiddleware(
        config={"max_visible_tools": 1, "top_k": 1},
        load_index_fn=load_index,
    )
    request = SimpleNamespace(
        tools=[native_tool, second_tool, *middleware.tools],
        runtime=SimpleNamespace(config={"configurable": {"thread_id": "thread-1"}}),
        messages=[HumanMessage("read a file", id="user-1")],
        system_prompt="Base prompt",
        override=lambda **kwargs: SimpleNamespace(**{**request.__dict__, **kwargs}),
    )

    modified = middleware._prepare_request(request)

    assert [tool.name for tool in modified.tools] == [
        "retrieve_tools",
        "call_retrieved_tool",
    ]
    assert "Tool retrieval is enabled" in modified.system_prompt


def test_prepare_request_uses_fallback_catalog_when_visible_tools_are_helpers() -> None:
    """Middleware ordering can expose only helper tools in `request.tools`."""
    native_tool = StructuredTool.from_function(
        _read_file, name="read_file", description="Read a file"
    )
    second_tool = StructuredTool.from_function(
        _echo_value, name="echo_value", description="Echo a value"
    )

    middleware = ToolRetrievalMiddleware(
        config={"top_k": 1},
        fallback_tools=[native_tool, second_tool],
        load_index_fn=lambda *_args, **_kwargs: ToolRetrievalIndex(
            index=object(),
            metadata={},
            entries=[{"name": "read_file"}, {"name": "echo_value"}],
            vector_dimensions=2,
            index_path=Path("/tmp/index.faiss"),
            metadata_path=Path("/tmp/index.meta.json"),
        ),
    )
    request = SimpleNamespace(
        tools=list(middleware.tools),
        runtime=SimpleNamespace(config={"configurable": {"thread_id": "thread-1"}}),
        messages=[HumanMessage("read a file", id="user-1")],
        system_prompt="Base prompt",
        override=lambda **kwargs: SimpleNamespace(**{**request.__dict__, **kwargs}),
    )

    modified = middleware._prepare_request(request)
    state = middleware._states["thread-1"]

    assert [tool.name for tool in modified.tools] == [
        "retrieve_tools",
        "call_retrieved_tool",
    ]
    assert sorted(state.records_by_name) == ["echo_value", "read_file"]


def test_retrieve_failure_returns_error_and_clears_state() -> None:
    def fail_select(*_args: object, **_kwargs: object) -> ToolRetrievalResult:
        msg = "embedding failed"
        raise ToolRetrievalError(msg)

    middleware = ToolRetrievalMiddleware(
        select_fn=fail_select,
        load_index_fn=lambda *_args, **_kwargs: ToolRetrievalIndex(
            index=object(),
            metadata={},
            entries=[{"name": "read_file"}],
            vector_dimensions=2,
            index_path=Path("/tmp/index.faiss"),
            metadata_path=Path("/tmp/index.meta.json"),
        ),
    )
    native_tool = StructuredTool.from_function(
        _read_file, name="read_file", description="Read a file"
    )
    state = _ThreadRetrievalState(
        turn_key="user-1",
        records_by_name={
            "read_file": next(record for record in _tool_records([native_tool]))
        },
        retrieved_names=["read_file"],
    )
    middleware._states["thread-1"] = state
    request = _tool_request("retrieve_tools", {"query": "read files"})

    result = middleware.wrap_tool_call(
        request, lambda _request: pytest.fail("handler should not run")
    )
    payload = json.loads(result.content)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "embedding failed" in payload["error"]
    assert state.retrieved_names == []


def test_retrieve_tools_uses_fallback_catalog_when_state_is_empty() -> None:
    """Tool calls may run on a fresh middleware state in the server runtime."""
    native_tool = StructuredTool.from_function(
        _read_file, name="read_file", description="Read a file"
    )

    def select(
        _tools: list[dict[str, object]],
        _query: str,
        _config: dict[str, object],
        **_kwargs: object,
    ) -> ToolRetrievalResult:
        return ToolRetrievalResult(
            selected_tools=[],
            selected_names=["read_file"],
            scores={"read_file": 0.9},
        )

    middleware = ToolRetrievalMiddleware(
        fallback_tools=[native_tool],
        select_fn=select,
        load_index_fn=lambda *_args, **_kwargs: ToolRetrievalIndex(
            index=object(),
            metadata={},
            entries=[{"name": "read_file"}],
            vector_dimensions=2,
            index_path=Path("/tmp/index.faiss"),
            metadata_path=Path("/tmp/index.meta.json"),
        ),
    )
    request = _tool_request("retrieve_tools", {"query": "read files"})

    result = middleware.wrap_tool_call(
        request, lambda _request: pytest.fail("handler should not run")
    )
    payload = json.loads(result.content)

    assert payload["success"] is True
    assert payload["retrieved_tools"] == ["read_file"]
    assert "read_file" in middleware._states["thread-1"].records_by_name


def test_call_retrieved_tool_dispatches_only_latest_retrieved_tool() -> None:
    native_tool = StructuredTool.from_function(
        _echo_value, name="echo_value", description="Echo a value"
    )
    middleware = ToolRetrievalMiddleware()
    record = next(record for record in _tool_records([native_tool]))
    middleware._states["thread-1"] = _ThreadRetrievalState(
        turn_key="user-1",
        records_by_name={"echo_value": record},
        retrieved_names=["echo_value"],
    )
    request = _tool_request(
        "call_retrieved_tool",
        {"name": "echo_value", "arguments": {"value": "ok"}},
    )

    def handler(native_request: ToolCallRequest) -> ToolMessage:
        assert native_request.tool_call["name"] == "echo_value"
        assert native_request.tool_call["args"] == {"value": "ok"}
        assert native_request.tool is native_tool
        return ToolMessage(
            content="ok", name="echo_value", tool_call_id=native_request.tool_call["id"]
        )

    result = middleware.wrap_tool_call(request, handler)

    assert result.content == "ok"


def test_call_retrieved_tool_rejects_unretrieved_tool() -> None:
    native_tool = StructuredTool.from_function(
        _echo_value, name="echo_value", description="Echo a value"
    )
    middleware = ToolRetrievalMiddleware()
    record = next(record for record in _tool_records([native_tool]))
    middleware._states["thread-1"] = _ThreadRetrievalState(
        turn_key="user-1",
        records_by_name={"echo_value": record},
        retrieved_names=["echo_value"],
    )
    request = _tool_request(
        "call_retrieved_tool",
        {"name": "other_tool", "arguments": {}},
    )

    result = middleware.wrap_tool_call(
        request, lambda _request: pytest.fail("handler should not run")
    )
    payload = json.loads(result.content)

    assert result.status == "error"
    assert "was not returned" in payload["error"]


def test_parse_args_accepts_no_retrieval_tool_call() -> None:
    with patch.object(
        sys, "argv", ["deepagents", "--no-retrieval-tool-call", "-n", "task"]
    ):
        args = parse_args()

    assert args.no_retrieval_tool_call is True


def _tool_request(name: str, args: dict[str, object]) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": name, "args": args, "id": "call-1", "type": "tool_call"},
        tool=None,
        state={},
        runtime=SimpleNamespace(config={"configurable": {"thread_id": "thread-1"}}),
    )


def _read_file(path: str) -> str:
    """Read a file."""
    return path


def _echo_value(value: str) -> str:
    """Echo a value."""
    return value
