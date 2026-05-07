"""Local embedding-backed tool retrieval for CLI-created agents."""

# ruff: noqa: ANN401,ARG001,B904,B905,BLE001,DOC201,DOC501,E501,EM101,PLR2004,PLR6104,PLR6301,PLW2901,SIM105,SIM110,TC002,TRY003,TRY301

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import math
import re
import threading
import uuid
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CACHE_DIR = Path.home() / ".deepagents" / "cache" / "tool_retrieval"
DEFAULT_MODEL_CACHE_DIR = DEFAULT_CACHE_DIR / "models"
DEFAULT_INDEX_BACKEND = "faiss"
DEFAULT_INDEX_TYPE = "flat_ip"
DEFAULT_TOP_K = 5
DEFAULT_MAX_VISIBLE_TOOLS = 16
DEFAULT_PLATFORM = "cli"

RETRIEVE_TOOLS_NAME = "retrieve_tools"
CALL_RETRIEVED_TOOL_NAME = "call_retrieved_tool"
RETRIEVAL_TOOL_NAMES = frozenset({RETRIEVE_TOOLS_NAME, CALL_RETRIEVED_TOOL_NAME})

TOOL_RETRIEVAL_SYSTEM_PROMPT = (
    "Tool retrieval is enabled. Most native tool schemas are hidden at first. "
    "When a task requires a tool, call `retrieve_tools` with a concise `query` "
    "describing the needed capability or next action. The retrieval tool will "
    "return matching native tool schemas for the current user turn; after it "
    "returns, call `call_retrieved_tool` with `name` set to a returned tool name "
    "and `arguments` matching that tool's schema. If a suitable tool was already "
    "retrieved in this user turn, call it through `call_retrieved_tool` instead "
    "of calling `retrieve_tools` again. Do not guess hidden tool names or arguments "
    "before retrieving them."
)


class ToolRetrievalError(RuntimeError):
    """Raised when tool retrieval cannot select a native tool subset."""


@dataclass
class ToolRetrievalResult:
    """Result returned by a retrieval query."""

    selected_tools: list[dict[str, Any]]
    selected_names: list[str]
    scores: dict[str, float]
    fallback_reason: str | None = None


@dataclass(frozen=True)
class ToolRetrievalPaths:
    """Disk artifact paths for one cached FAISS index."""

    index_path: Path
    metadata_path: Path


@dataclass
class ToolRetrievalIndex:
    """Prepared FAISS index and matching tool metadata."""

    index: Any
    metadata: dict[str, Any]
    entries: list[dict[str, Any]]
    vector_dimensions: int
    index_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class _ToolRecord:
    name: str
    schema: dict[str, Any]
    tool: BaseTool | dict[str, Any]


@dataclass
class _ThreadRetrievalState:
    turn_key: str | None = None
    records_by_name: dict[str, _ToolRecord] | None = None
    retrieved_names: list[str] | None = None


Embedder = Callable[[list[str], dict[str, Any]], Any]

_MODEL_CACHE: dict[tuple[str, str, str, str, bool], Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def default_tool_retrieval_config() -> dict[str, Any]:
    """Return Hermes-compatible local retrieval defaults for Deep Agents CLI."""
    return {
        "model": DEFAULT_EMBEDDING_MODEL,
        "device": "cpu",
        "normalize_embeddings": True,
        "index_backend": DEFAULT_INDEX_BACKEND,
        "index_type": DEFAULT_INDEX_TYPE,
        "top_k": DEFAULT_TOP_K,
        "max_visible_tools": DEFAULT_MAX_VISIBLE_TOOLS,
        "cache_dir": str(DEFAULT_CACHE_DIR),
        "model_cache_dir": str(DEFAULT_MODEL_CACHE_DIR),
    }


def tool_name(tool: dict[str, Any]) -> str:
    """Return the function name from an OpenAI-style tool schema."""
    return str((tool or {}).get("function", {}).get("name") or "")


def _schema_property_lines(
    properties: dict[str, Any],
    *,
    prefix: str = "",
    required: Iterable[str] = (),
) -> list[str]:
    lines: list[str] = []
    required_set = set(required or ())
    for prop_name in sorted(properties):
        prop = properties.get(prop_name)
        if not isinstance(prop, dict):
            continue
        path = f"{prefix}.{prop_name}" if prefix else prop_name
        type_text = prop.get("type") or prop.get("format") or ""
        if isinstance(type_text, list):
            type_text = "|".join(str(value) for value in type_text)
        desc = str(prop.get("description") or "").strip()
        marker = " required" if prop_name in required_set else ""
        lines.append(f"parameter {path}{marker}: {type_text} {desc}".strip())

        nested = prop.get("properties")
        if isinstance(nested, dict):
            lines.extend(
                _schema_property_lines(
                    nested,
                    prefix=path,
                    required=prop.get("required") or (),
                )
            )
        items = prop.get("items")
        if isinstance(items, dict) and isinstance(items.get("properties"), dict):
            lines.extend(
                _schema_property_lines(
                    items["properties"],
                    prefix=f"{path}[]",
                    required=items.get("required") or (),
                )
            )
    return lines


def tool_schema_text(tool: dict[str, Any]) -> str:
    """Build stable embedding text from a tool schema."""
    function = (tool or {}).get("function") or {}
    name = str(function.get("name") or "")
    parts = [
        f"name: {name}",
        f"description: {str(function.get('description') or '').strip()}",
    ]
    params = function.get("parameters") or {}
    if isinstance(params, dict):
        properties = params.get("properties") or {}
        if isinstance(properties, dict):
            parts.extend(
                _schema_property_lines(
                    properties,
                    required=params.get("required") or (),
                )
            )
    return "\n".join(part for part in parts if part.strip())


def tool_schema_hash(tools: Sequence[dict[str, Any]]) -> str:
    """Fingerprint complete tool schemas for cache invalidation."""
    payload = [
        {"name": tool_name(tool), "function": (tool or {}).get("function") or {}}
        for tool in sorted(tools or [], key=tool_name)
    ]
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _is_truthy(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off", "never"}:
            return False
        return default
    return bool(value)


def _model_name(config: dict[str, Any]) -> str:
    return (
        str(config.get("model") or DEFAULT_EMBEDDING_MODEL).strip()
        or DEFAULT_EMBEDDING_MODEL
    )


def _device(config: dict[str, Any]) -> str:
    return str(config.get("device") or "cpu").strip() or "cpu"


def _revision(config: dict[str, Any]) -> str:
    return str(config.get("revision") or "").strip()


def _normalize_embeddings(config: dict[str, Any]) -> bool:
    return _is_truthy(config.get("normalize_embeddings", True), default=True)


def _top_k(config: dict[str, Any]) -> int:
    try:
        top_k = int(config.get("top_k", DEFAULT_TOP_K))
    except (TypeError, ValueError):
        msg = "top_k must be an integer"
        raise ToolRetrievalError(msg) from None
    if top_k <= 0:
        msg = "top_k must be > 0"
        raise ToolRetrievalError(msg)
    return top_k


def _max_visible_tools(config: dict[str, Any], total_native_tools: int) -> int:
    try:
        max_tools = int(config.get("max_visible_tools", DEFAULT_MAX_VISIBLE_TOOLS))
    except (TypeError, ValueError):
        max_tools = DEFAULT_MAX_VISIBLE_TOOLS
    max_tools = max(1, max_tools)
    return min(max_tools, max(1, int(total_native_tools or 1)))


def _validate_index_config(config: dict[str, Any]) -> None:
    _top_k(config)
    if (
        str(config.get("index_backend") or DEFAULT_INDEX_BACKEND)
        != DEFAULT_INDEX_BACKEND
    ):
        msg = "only the faiss tool retrieval index backend is supported"
        raise ToolRetrievalError(msg)
    if str(config.get("index_type") or DEFAULT_INDEX_TYPE) != DEFAULT_INDEX_TYPE:
        msg = "only the flat_ip FAISS tool retrieval index type is supported"
        raise ToolRetrievalError(msg)


def _cache_dir(config: dict[str, Any]) -> Path:
    raw_dir = config.get("cache_dir") or str(DEFAULT_CACHE_DIR)
    cache_dir = Path(str(raw_dir)).expanduser()
    if not cache_dir.is_absolute():
        cache_dir = Path.home() / ".deepagents" / cache_dir
    return cache_dir


def _model_cache_dir(config: dict[str, Any]) -> Path:
    raw_dir = config.get("model_cache_dir") or str(DEFAULT_MODEL_CACHE_DIR)
    cache_dir = Path(str(raw_dir)).expanduser()
    if not cache_dir.is_absolute():
        cache_dir = Path.home() / ".deepagents" / cache_dir
    return cache_dir


def _slug(value: str, default: str = "default") -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = text.strip(".-_")
    return text or default


def index_artifact_paths(
    config: dict[str, Any],
    schema_hash: str,
    platform: str | None = None,
) -> ToolRetrievalPaths:
    """Return platform- and model-scoped FAISS artifact paths."""
    model_slug = _slug(_model_name(config), "model")
    platform_slug = _slug(platform or DEFAULT_PLATFORM, DEFAULT_PLATFORM)
    stem = _slug(schema_hash, "index")
    directory = _cache_dir(config) / platform_slug / model_slug
    return ToolRetrievalPaths(
        index_path=directory / f"{stem}.faiss",
        metadata_path=directory / f"{stem}.meta.json",
    )


def _import_faiss() -> Any:
    try:
        import faiss  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on install env
        msg = "faiss package unavailable; install faiss-cpu to use local tool retrieval"
        raise ToolRetrievalError(msg) from exc
    return faiss


def _import_numpy() -> Any:
    try:
        import numpy as np  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on install env
        msg = "numpy package unavailable; install numpy to use local tool retrieval"
        raise ToolRetrievalError(msg) from exc
    return np


def _get_sentence_transformer_model(config: dict[str, Any]) -> Any:
    model_name = _model_name(config)
    device = _device(config)
    revision = _revision(config)
    cache_folder = str(_model_cache_dir(config))
    local_files_only = _is_truthy(config.get("local_files_only", False), default=False)
    key = (model_name, device, cache_folder, revision, local_files_only)

    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached

        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - depends on install env
            msg = (
                "sentence-transformers package unavailable; install sentence-transformers "
                f"to use local tool retrieval: {type(exc).__name__}: {exc}"
            )
            raise ToolRetrievalError(msg) from exc

        try:
            kwargs: dict[str, Any] = {
                "device": device,
                "cache_folder": cache_folder,
                "local_files_only": local_files_only,
            }
            if revision:
                kwargs["revision"] = revision
            model = SentenceTransformer(model_name, **kwargs)
        except Exception as exc:
            msg = f"failed to load local embedding model '{model_name}' on device '{device}': {exc}"
            raise ToolRetrievalError(msg) from exc

        _MODEL_CACHE[key] = model
        return model


def _python_matrix(vectors: Any) -> list[list[float]]:
    if hasattr(vectors, "tolist"):
        vectors = vectors.tolist()
    if isinstance(vectors, tuple):
        vectors = list(vectors)
    if not isinstance(vectors, list):
        msg = "embedding output was not a matrix"
        raise ToolRetrievalError(msg)
    if vectors and all(not isinstance(item, (list, tuple)) for item in vectors):
        vectors = [vectors]

    matrix: list[list[float]] = []
    for row in vectors:
        if hasattr(row, "tolist"):
            row = row.tolist()
        if not isinstance(row, (list, tuple)):
            msg = "embedding row was not a vector"
            raise ToolRetrievalError(msg)
        try:
            values = [float(item) for item in row]
        except (TypeError, ValueError) as exc:
            msg = f"embedding row contained non-numeric values: {exc}"
            raise ToolRetrievalError(msg) from exc
        if not values:
            msg = "empty embedding vector"
            raise ToolRetrievalError(msg)
        if not all(math.isfinite(value) for value in values):
            msg = "embedding vector contained non-finite values"
            raise ToolRetrievalError(msg)
        matrix.append(values)
    return matrix


def _coerce_embedding_matrix(
    vectors: Any,
    *,
    expected_count: int,
    normalize: bool,
    require_numpy: bool = False,
) -> Any:
    try:
        np = _import_numpy()
    except ToolRetrievalError:
        if require_numpy:
            raise
        matrix = _python_matrix(vectors)
        if len(matrix) != expected_count:
            msg = f"embedding count did not match input count: got {len(matrix)}, expected {expected_count}"
            raise ToolRetrievalError(msg)
        width = len(matrix[0]) if matrix else 0
        if width <= 0 or any(len(row) != width for row in matrix):
            msg = "embedding dimensions were empty or inconsistent"
            raise ToolRetrievalError(msg)
        if normalize:
            normalized = []
            for row in matrix:
                norm = math.sqrt(sum(value * value for value in row))
                if norm == 0.0:
                    msg = "embedding vector had zero norm"
                    raise ToolRetrievalError(msg)
                normalized.append([value / norm for value in row])
            matrix = normalized
        return matrix

    try:
        array = np.asarray(vectors, dtype=np.float32)
    except Exception as exc:
        msg = f"embedding output could not be converted to float32: {exc}"
        raise ToolRetrievalError(msg) from exc
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        msg = f"embedding output must be 2D, got shape {getattr(array, 'shape', None)}"
        raise ToolRetrievalError(msg)
    if array.shape[0] != expected_count:
        msg = f"embedding count did not match input count: got {array.shape[0]}, expected {expected_count}"
        raise ToolRetrievalError(msg)
    if array.shape[1] <= 0:
        msg = "empty embedding vector"
        raise ToolRetrievalError(msg)
    if not np.isfinite(array).all():
        msg = "embedding vector contained non-finite values"
        raise ToolRetrievalError(msg)
    if normalize:
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        if (norms == 0.0).any():
            msg = "embedding vector had zero norm"
            raise ToolRetrievalError(msg)
        array = array / norms
    return np.ascontiguousarray(array, dtype=np.float32)


def _matrix_width(matrix: Any) -> int:
    shape = getattr(matrix, "shape", None)
    if shape is not None and len(shape) == 2:
        return int(shape[1])
    if isinstance(matrix, list) and matrix and isinstance(matrix[0], list):
        return len(matrix[0])
    return 0


def embed_texts_local(texts: list[str], config: dict[str, Any]) -> Any:
    """Embed text with a cached local SentenceTransformer model."""
    if not texts:
        return []
    model = _get_sentence_transformer_model(config or {})
    try:
        batch_size = int(config.get("batch_size", 32) or 32)
    except (TypeError, ValueError):
        batch_size = 32
    normalize = _normalize_embeddings(config or {})

    try:
        vectors = model.encode(
            texts,
            batch_size=max(1, batch_size),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
    except Exception as exc:
        msg = f"local embedding model failed to encode text: {exc}"
        raise ToolRetrievalError(msg) from exc
    return _coerce_embedding_matrix(
        vectors,
        expected_count=len(texts),
        normalize=normalize,
        require_numpy=True,
    )


def preload_embedding_model(config: dict[str, Any]) -> None:
    """Load the configured local embedding model into the process cache."""
    _import_numpy()
    _get_sentence_transformer_model(config or {})


def _tool_entries(tools: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for tool in tools or []:
        name = tool_name(tool)
        if not name:
            continue
        text = tool_schema_text(tool)
        entries.append(
            {
                "name": name,
                "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )
    return entries


def _index_metadata(
    tools: Sequence[dict[str, Any]],
    config: dict[str, Any],
    platform: str | None,
) -> dict[str, Any]:
    entries = _tool_entries(tools)
    model = _model_name(config)
    return {
        "schema_hash": tool_schema_hash(tools),
        "model": model,
        "embedding_model": model,
        "model_revision": _revision(config),
        "device": _device(config),
        "backend": "sentence-transformers",
        "index_backend": str(config.get("index_backend") or DEFAULT_INDEX_BACKEND),
        "index_type": str(config.get("index_type") or DEFAULT_INDEX_TYPE),
        "normalize_embeddings": _normalize_embeddings(config),
        "platform": platform or "",
        "tool_count": len(entries),
        "tool_names": [entry["name"] for entry in entries],
        "tool_text_hashes": [entry["text_hash"] for entry in entries],
        "entries": entries,
    }


def _load_metadata(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as exc:
        msg = f"tool retrieval metadata is corrupted at {path}: {exc}"
        raise ToolRetrievalError(msg) from exc
    if not isinstance(data, dict):
        msg = f"tool retrieval metadata is not an object at {path}"
        raise ToolRetrievalError(msg)
    return data


def _metadata_matches(metadata: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key, value in expected.items():
        if metadata.get(key) != value:
            return False
    return True


def _load_index(
    paths: ToolRetrievalPaths,
    expected_metadata: dict[str, Any],
) -> ToolRetrievalIndex | None:
    metadata = _load_metadata(paths.metadata_path)
    if metadata is None or not paths.index_path.exists():
        return None
    if not _metadata_matches(metadata, expected_metadata):
        return None

    entries = metadata.get("entries")
    if not isinstance(entries, list) or not entries:
        msg = f"tool retrieval metadata has no entries at {paths.metadata_path}"
        raise ToolRetrievalError(msg)
    vector_dimensions = metadata.get("vector_dimensions")
    if not isinstance(vector_dimensions, int) or vector_dimensions <= 0:
        msg = f"tool retrieval metadata has invalid vector_dimensions at {paths.metadata_path}"
        raise ToolRetrievalError(msg)

    faiss = _import_faiss()
    try:
        index = faiss.read_index(str(paths.index_path))
    except Exception as exc:
        msg = f"tool retrieval FAISS index is corrupted at {paths.index_path}: {exc}"
        raise ToolRetrievalError(msg) from exc

    index_dim = int(getattr(index, "d", 0) or 0)
    index_total = int(getattr(index, "ntotal", 0) or 0)
    if index_dim != vector_dimensions:
        msg = f"tool retrieval FAISS dimension mismatch: index has {index_dim}, metadata has {vector_dimensions}"
        raise ToolRetrievalError(msg)
    if index_total != len(entries):
        msg = f"tool retrieval FAISS entry count mismatch: index has {index_total}, metadata has {len(entries)}"
        raise ToolRetrievalError(msg)
    return ToolRetrievalIndex(
        index=index,
        metadata=metadata,
        entries=entries,
        vector_dimensions=vector_dimensions,
        index_path=paths.index_path,
        metadata_path=paths.metadata_path,
    )


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp_path.write_text(
            json.dumps(
                data,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ),
            encoding="utf-8",
        )
        tmp_path.replace(path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            logger.debug(
                "Failed to remove temporary metadata file %s", tmp_path, exc_info=True
            )


def _write_faiss_index_atomic(faiss: Any, index: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        faiss.write_index(index, str(tmp_path))
        tmp_path.replace(path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            logger.debug(
                "Failed to remove temporary FAISS index file %s",
                tmp_path,
                exc_info=True,
            )


def _build_index(
    tools: Sequence[dict[str, Any]],
    config: dict[str, Any],
    platform: str | None,
    *,
    embedder: Embedder,
    expected_metadata: dict[str, Any],
    paths: ToolRetrievalPaths,
) -> ToolRetrievalIndex:
    _validate_index_config(config)

    tool_texts: list[str] = []
    entries: list[dict[str, str]] = []
    for tool in tools:
        name = tool_name(tool)
        if not name:
            continue
        text = tool_schema_text(tool)
        tool_texts.append(text)
        entries.append(
            {
                "name": name,
                "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )

    if not entries:
        msg = "no tool schemas available for local retrieval"
        raise ToolRetrievalError(msg)

    vectors = embedder(tool_texts, config)
    matrix = _coerce_embedding_matrix(
        vectors,
        expected_count=len(tool_texts),
        normalize=_normalize_embeddings(config),
    )
    vector_dimensions = _matrix_width(matrix)
    if vector_dimensions <= 0:
        msg = "empty embedding vector"
        raise ToolRetrievalError(msg)

    faiss = _import_faiss()
    try:
        index = faiss.IndexFlatIP(vector_dimensions)
        index.add(matrix)
    except Exception as exc:
        msg = f"failed to build FAISS tool retrieval index: {exc}"
        raise ToolRetrievalError(msg) from exc
    if int(getattr(index, "ntotal", 0) or 0) != len(entries):
        msg = "FAISS index did not store all tool embeddings"
        raise ToolRetrievalError(msg)

    metadata = dict(expected_metadata)
    metadata["vector_dimensions"] = vector_dimensions
    metadata["entries"] = entries

    _write_faiss_index_atomic(faiss, index, paths.index_path)
    _write_json_atomic(paths.metadata_path, metadata)
    return ToolRetrievalIndex(
        index=index,
        metadata=metadata,
        entries=entries,
        vector_dimensions=vector_dimensions,
        index_path=paths.index_path,
        metadata_path=paths.metadata_path,
    )


def load_or_build_index(
    tools: Sequence[dict[str, Any]],
    config: dict[str, Any],
    platform: str | None = None,
    *,
    embedder: Embedder | None = None,
) -> ToolRetrievalIndex:
    """Load a cached FAISS index or build one for the given tool schemas."""
    if not tools:
        msg = "no tools available"
        raise ToolRetrievalError(msg)
    _validate_index_config(config)
    if embedder is None:
        preload_embedding_model(config)
        embedder = embed_texts_local
    expected_metadata = _index_metadata(tools, config, platform)
    paths = index_artifact_paths(config, expected_metadata["schema_hash"], platform)

    cache_error: ToolRetrievalError | None = None
    try:
        cached = _load_index(paths, expected_metadata)
    except ToolRetrievalError as exc:
        cache_error = exc
        logger.warning("Tool retrieval cache invalid; rebuilding: %s", exc)
    else:
        if cached is not None:
            return cached

    try:
        return _build_index(
            tools,
            config,
            platform,
            embedder=embedder,
            expected_metadata=expected_metadata,
            paths=paths,
        )
    except ToolRetrievalError as exc:
        if cache_error is not None:
            msg = f"failed to rebuild tool retrieval index after cache failure ({cache_error}): {exc}"
            raise ToolRetrievalError(msg) from exc
        raise


def select_tools_for_query(
    tools: Sequence[dict[str, Any]],
    query: str,
    config: dict[str, Any],
    platform: str | None = None,
    *,
    embedder: Embedder | None = None,
    prepared_index: ToolRetrievalIndex | None = None,
) -> ToolRetrievalResult:
    """Return the top-K native tool schemas for a query."""
    if not tools:
        msg = "no tools available"
        raise ToolRetrievalError(msg)
    query = (query or "").strip()
    if not query:
        msg = "tool retrieval query is empty"
        raise ToolRetrievalError(msg)

    _validate_index_config(config)
    top_k = _top_k(config)
    embedder = embedder or embed_texts_local
    retrieval_index = prepared_index or load_or_build_index(
        tools,
        config,
        platform,
        embedder=embedder,
    )

    query_vectors = embedder([query], config)
    query_matrix = _coerce_embedding_matrix(
        query_vectors,
        expected_count=1,
        normalize=_normalize_embeddings(config),
    )
    if _matrix_width(query_matrix) != retrieval_index.vector_dimensions:
        msg = (
            f"query embedding dimension mismatch: got {_matrix_width(query_matrix)}, "
            f"expected {retrieval_index.vector_dimensions}"
        )
        raise ToolRetrievalError(msg)

    k = min(top_k, len(retrieval_index.entries))
    try:
        distances, indices = retrieval_index.index.search(query_matrix, k)
    except Exception as exc:
        msg = f"FAISS tool retrieval search failed: {exc}"
        raise ToolRetrievalError(msg) from exc

    if hasattr(distances, "tolist"):
        distances = distances.tolist()
    if hasattr(indices, "tolist"):
        indices = indices.tolist()
    score_row = distances[0] if distances else []
    index_row = indices[0] if indices else []

    selected_names: list[str] = []
    scores: dict[str, float] = {}
    for raw_score, raw_idx in zip(score_row, index_row):
        idx = int(raw_idx)
        if idx < 0 or idx >= len(retrieval_index.entries):
            continue
        name = str(retrieval_index.entries[idx].get("name") or "")
        if not name:
            continue
        selected_names.append(name)
        scores[name] = float(raw_score)

    if not selected_names:
        msg = "no indexed tool scores available"
        raise ToolRetrievalError(msg)

    by_name = {tool_name(tool): tool for tool in tools}
    selected_tools = [by_name[name] for name in selected_names if name in by_name]
    if not selected_tools:
        msg = "retrieval returned no usable tool schemas"
        raise ToolRetrievalError(msg)
    return ToolRetrievalResult(
        selected_tools=selected_tools,
        selected_names=[tool_name(tool) for tool in selected_tools],
        scores=scores,
    )


def _parameters_from_base_tool(tool: BaseTool) -> dict[str, Any]:
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is not None and hasattr(args_schema, "model_json_schema"):
        try:
            schema = args_schema.model_json_schema()
            if isinstance(schema, dict):
                schema.setdefault("type", "object")
                schema.setdefault("properties", {})
                return copy.deepcopy(schema)
        except Exception:
            logger.debug(
                "Failed to render args_schema for tool %s", tool.name, exc_info=True
            )

    args = getattr(tool, "args", None)
    if isinstance(args, dict):
        return {
            "type": "object",
            "properties": copy.deepcopy(args),
        }
    return {"type": "object", "properties": {}}


def _schema_from_base_tool(tool: BaseTool) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": str(getattr(tool, "description", "") or ""),
            "parameters": _parameters_from_base_tool(tool),
        },
    }


def _schema_from_dict_tool(tool: dict[str, Any]) -> dict[str, Any] | None:
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        function = copy.deepcopy(tool["function"])
        if not function.get("name"):
            return None
        function.setdefault("description", "")
        function.setdefault("parameters", {"type": "object", "properties": {}})
        return {"type": "function", "function": function}
    if tool.get("name"):
        return {
            "type": "function",
            "function": {
                "name": str(tool["name"]),
                "description": str(tool.get("description") or ""),
                "parameters": copy.deepcopy(
                    tool.get("parameters") or {"type": "object", "properties": {}}
                ),
            },
        }
    return None


def _tool_records(tools: Sequence[BaseTool | dict[str, Any]]) -> list[_ToolRecord]:
    records: list[_ToolRecord] = []
    seen: set[str] = set()
    for tool in tools or []:
        if isinstance(tool, BaseTool):
            schema = _schema_from_base_tool(tool)
        elif isinstance(tool, dict):
            schema = _schema_from_dict_tool(tool)
            if schema is None:
                continue
        else:
            continue

        name = tool_name(schema)
        if not name or name in RETRIEVAL_TOOL_NAMES or name in seen:
            continue
        seen.add(name)
        records.append(_ToolRecord(name=name, schema=schema, tool=tool))
    return records


def _latest_user_turn_key(messages: Sequence[BaseMessage]) -> str:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        role = getattr(message, "type", None) or getattr(message, "role", None)
        if role not in {"human", "user"}:
            continue
        message_id = getattr(message, "id", None)
        if message_id:
            return str(message_id)
        content = getattr(message, "content", "")
        digest = hashlib.sha256(
            str(content).encode("utf-8", errors="replace")
        ).hexdigest()
        return f"{index}:{digest}"
    digest = hashlib.sha256(
        repr(messages).encode("utf-8", errors="replace")
    ).hexdigest()
    return f"no-user:{digest}"


def _thread_key_from_runtime(runtime: Any) -> str:
    config = getattr(runtime, "config", None)
    if isinstance(config, dict):
        configurable = config.get("configurable")
        if isinstance(configurable, dict):
            thread_id = configurable.get("thread_id") or configurable.get(
                "checkpoint_id"
            )
            if thread_id:
                return str(thread_id)
    return "default"


def _make_retrieve_tools_tool() -> StructuredTool:
    def retrieve_tools(query: str) -> str:
        """Retrieve native Deep Agents tool schemas for the current user turn."""
        return json.dumps(
            {
                "success": False,
                "error": "retrieve_tools must be handled by ToolRetrievalMiddleware",
            },
            ensure_ascii=False,
        )

    return StructuredTool.from_function(
        retrieve_tools,
        name=RETRIEVE_TOOLS_NAME,
        description=(
            "Retrieve native Deep Agents tool schemas for the current user turn. "
            "Use this before calling call_retrieved_tool."
        ),
    )


def _make_call_retrieved_tool() -> StructuredTool:
    def call_retrieved_tool(name: str, arguments: dict[str, Any]) -> str:
        """Call a native Deep Agents tool returned by retrieve_tools."""
        return json.dumps(
            {
                "success": False,
                "error": "call_retrieved_tool must be handled by ToolRetrievalMiddleware",
                "name": name,
                "arguments": arguments,
            },
            ensure_ascii=False,
        )

    return StructuredTool.from_function(
        call_retrieved_tool,
        name=CALL_RETRIEVED_TOOL_NAME,
        description=(
            "Call a native Deep Agents tool returned by retrieve_tools. "
            "The name must match one of the latest retrieved tool schemas."
        ),
    )


class ToolRetrievalMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    """Expose model-called retrieval helpers instead of the full native tool catalog."""

    tools: Sequence[BaseTool]

    def __init__(
        self,
        *,
        config: dict[str, Any] | None = None,
        platform: str = DEFAULT_PLATFORM,
        fallback_tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
        select_fn: Callable[..., ToolRetrievalResult] = select_tools_for_query,
        load_index_fn: Callable[..., ToolRetrievalIndex] = load_or_build_index,
    ) -> None:
        """Initialize the middleware.

        Args:
            config: Retrieval configuration. Missing values use Hermes defaults.
            platform: Cache namespace for index artifacts.
            fallback_tools: Native tool catalog used when middleware ordering
                exposes only retrieval helpers in `request.tools`.
            select_fn: Retrieval function, injectable for tests.
            load_index_fn: Index loading function, injectable for tests.
        """
        super().__init__()
        merged = default_tool_retrieval_config()
        if config:
            merged.update(config)
        self.config = merged
        self.platform = platform
        self.tools = [_make_retrieve_tools_tool(), _make_call_retrieved_tool()]
        self.fallback_tools = list(fallback_tools or [])
        self._select_fn = select_fn
        self._load_index_fn = load_index_fn
        self._states: dict[str, _ThreadRetrievalState] = {}
        self._index_cache: dict[str, ToolRetrievalIndex] = {}
        self._lock = threading.RLock()

    def _state_for_runtime(self, runtime: Any) -> _ThreadRetrievalState:
        thread_key = _thread_key_from_runtime(runtime)
        with self._lock:
            state = self._states.get(thread_key)
            if state is None:
                state = _ThreadRetrievalState(records_by_name={}, retrieved_names=[])
                self._states[thread_key] = state
            return state

    def _ensure_index(self, records: Sequence[_ToolRecord]) -> ToolRetrievalIndex:
        schemas = [record.schema for record in records]
        schema_hash = tool_schema_hash(schemas)
        cache_key = "|".join(
            [
                self.platform,
                _model_name(self.config),
                _device(self.config),
                _revision(self.config),
                str(_normalize_embeddings(self.config)),
                schema_hash,
            ]
        )
        with self._lock:
            cached = self._index_cache.get(cache_key)
            if cached is not None:
                return cached
            index = self._load_index_fn(schemas, self.config, platform=self.platform)
            self._index_cache[cache_key] = index
            return index

    def _prepare_request(
        self, request: ModelRequest[ContextT]
    ) -> ModelRequest[ContextT]:
        records = _tool_records(request.tools or [])
        if not records and self.fallback_tools:
            records = _tool_records(self.fallback_tools)
        if not records:
            return request.override(tools=[])
        if len(records) <= _top_k(self.config):
            return request.override(tools=[record.tool for record in records])

        state = self._state_for_runtime(request.runtime)
        turn_key = _latest_user_turn_key(request.messages)
        with self._lock:
            if state.turn_key != turn_key:
                state.turn_key = turn_key
                state.retrieved_names = []
            state.records_by_name = {record.name: record for record in records}

        try:
            self._ensure_index(records)
        except Exception as exc:
            msg = f"tool retrieval initialization failed: {exc}"
            raise RuntimeError(msg) from exc

        system_prompt = request.system_prompt or ""
        if TOOL_RETRIEVAL_SYSTEM_PROMPT not in system_prompt:
            system_prompt = (
                f"{system_prompt}\n\n{TOOL_RETRIEVAL_SYSTEM_PROMPT}"
                if system_prompt
                else TOOL_RETRIEVAL_SYSTEM_PROMPT
            )
        return request.override(
            tools=list(self.tools),
            system_prompt=system_prompt,
            tool_choice=None,
        )

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Expose only retrieval helper tools to the model."""
        return handler(self._prepare_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[
            [ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]
        ],
    ) -> ModelResponse[ResponseT]:
        """Async variant of `wrap_model_call`."""
        prepared = await asyncio.to_thread(self._prepare_request, request)
        return await handler(prepared)

    def _tool_message(
        self,
        request: ToolCallRequest,
        payload: dict[str, Any],
        *,
        status: str = "success",
    ) -> ToolMessage:
        return ToolMessage(
            content=json.dumps(payload, ensure_ascii=False),
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
            status=status,
        )

    @staticmethod
    def _retrieved_tool_response_schema(
        schema: dict[str, Any], score: Any = None
    ) -> dict[str, Any]:
        function = schema.get("function", {}) if isinstance(schema, dict) else {}
        item: dict[str, Any] = {
            "name": function.get("name", ""),
            "description": function.get("description", ""),
            "parameters": copy.deepcopy(
                function.get("parameters") or {"type": "object", "properties": {}}
            ),
        }
        if score is not None:
            try:
                item["score"] = float(score)
            except (TypeError, ValueError):
                pass
        return item

    def _retrieve_tools(self, request: ToolCallRequest) -> ToolMessage:
        state = self._state_for_runtime(request.runtime)
        args = request.tool_call.get("args") or {}
        query = str(args.get("query") or "").strip()
        if not query:
            with self._lock:
                state.retrieved_names = []
            return self._tool_message(
                request,
                {
                    "success": False,
                    "error": "query is required",
                    "message": "Call retrieve_tools again with a concise capability query.",
                },
                status="error",
            )

        records_by_name = state.records_by_name or {}
        records = list(records_by_name.values())
        if not records:
            return self._tool_message(
                request,
                {
                    "success": False,
                    "error": "tool retrieval is enabled but no native tool schemas are available",
                },
                status="error",
            )

        schemas = [record.schema for record in records]
        try:
            prepared_index = self._ensure_index(records)
            result = self._select_fn(
                schemas,
                query,
                self.config,
                platform=self.platform,
                prepared_index=prepared_index,
            )
            if result.fallback_reason:
                raise RuntimeError(result.fallback_reason)
            selected_names = [
                name for name in result.selected_names if name in records_by_name
            ]
            if not selected_names:
                raise RuntimeError("retrieval returned no usable schemas")
            max_visible_tools = _max_visible_tools(self.config, len(records))
            selected_names = selected_names[:max_visible_tools]

            with self._lock:
                state.retrieved_names = list(selected_names)

            returned_tools = [
                self._retrieved_tool_response_schema(
                    records_by_name[name].schema,
                    result.scores.get(name),
                )
                for name in selected_names
            ]
            logger.debug(
                "Tool retrieval selected %s for platform=%s query=%r",
                selected_names,
                self.platform,
                query[:200],
            )
            return self._tool_message(
                request,
                {
                    "success": True,
                    "query": query,
                    "exposed_tools": selected_names,
                    "retrieved_tools": selected_names,
                    "tools": returned_tools,
                    "message": (
                        "These native tool schemas are available for this user turn. "
                        "Call call_retrieved_tool with name set to one of retrieved_tools "
                        "and arguments matching that tool's parameters. Call retrieve_tools "
                        "again only when you need a different capability."
                    ),
                },
            )
        except Exception as exc:
            with self._lock:
                state.retrieved_names = []
            logger.warning(
                "Tool retrieval failed for platform=%s: %s", self.platform, exc
            )
            return self._tool_message(
                request,
                {
                    "success": False,
                    "query": query,
                    "error": f"tool retrieval failed: {exc}",
                    "message": (
                        "No native tools were retrieved. Call retrieve_tools again with a different "
                        "query if you still need a hidden tool."
                    ),
                },
                status="error",
            )

    def _parse_retrieved_tool_args(
        self, request: ToolCallRequest
    ) -> tuple[str, dict[str, Any], ToolMessage | None]:
        args = request.tool_call.get("args") or {}
        requested_name = str(args.get("name") or "").strip()
        if not requested_name:
            return (
                "",
                {},
                self._tool_message(
                    request,
                    {
                        "success": False,
                        "error": "name is required",
                        "message": "Call retrieve_tools first, then call_retrieved_tool with a returned tool name.",
                    },
                    status="error",
                ),
            )

        raw_arguments = args.get("arguments", {})
        if raw_arguments is None:
            return requested_name, {}, None
        if isinstance(raw_arguments, dict):
            return requested_name, raw_arguments, None
        if isinstance(raw_arguments, str):
            try:
                parsed = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                return (
                    requested_name,
                    {},
                    self._tool_message(
                        request,
                        {
                            "success": False,
                            "error": f"arguments must be a JSON object: {exc}",
                        },
                        status="error",
                    ),
                )
            if isinstance(parsed, dict):
                return requested_name, parsed, None
        return (
            requested_name,
            {},
            self._tool_message(
                request,
                {
                    "success": False,
                    "error": "arguments must be an object",
                },
                status="error",
            ),
        )

    def _native_tool_request(
        self, request: ToolCallRequest
    ) -> tuple[ToolCallRequest | None, ToolMessage | None]:
        requested_name, native_args, error = self._parse_retrieved_tool_args(request)
        if error is not None:
            return None, error

        state = self._state_for_runtime(request.runtime)
        retrieved_names = list(state.retrieved_names or [])
        if not retrieved_names:
            return (
                None,
                self._tool_message(
                    request,
                    {
                        "success": False,
                        "error": "no retrieved tools are available",
                        "message": "Call retrieve_tools with the needed capability before call_retrieved_tool.",
                    },
                    status="error",
                ),
            )
        if requested_name not in retrieved_names:
            return (
                None,
                self._tool_message(
                    request,
                    {
                        "success": False,
                        "error": f"tool '{requested_name}' was not returned by the latest retrieve_tools call",
                        "retrieved_tools": retrieved_names,
                        "message": "Call retrieve_tools again if you need a different native tool.",
                    },
                    status="error",
                ),
            )

        record = (state.records_by_name or {}).get(requested_name)
        if record is None:
            return (
                None,
                self._tool_message(
                    request,
                    {
                        "success": False,
                        "error": f"tool '{requested_name}' is not in the native tool catalog",
                    },
                    status="error",
                ),
            )
        if not isinstance(record.tool, BaseTool):
            return (
                None,
                self._tool_message(
                    request,
                    {
                        "success": False,
                        "error": f"tool '{requested_name}' is not executable by the local tool dispatcher",
                    },
                    status="error",
                ),
            )

        native_call = {
            **request.tool_call,
            "name": requested_name,
            "args": native_args,
        }
        return request.override(tool_call=native_call, tool=record.tool), None

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Handle retrieval helpers or dispatch a retrieved native tool."""
        tool_name_in_call = request.tool_call["name"]
        if tool_name_in_call == RETRIEVE_TOOLS_NAME:
            return self._retrieve_tools(request)
        if tool_name_in_call == CALL_RETRIEVED_TOOL_NAME:
            native_request, error = self._native_tool_request(request)
            if error is not None:
                return error
            if native_request is None:
                return self._tool_message(
                    request,
                    {
                        "success": False,
                        "error": "failed to prepare native tool request",
                    },
                    status="error",
                )
            return handler(native_request)
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Async variant of `wrap_tool_call`."""
        tool_name_in_call = request.tool_call["name"]
        if tool_name_in_call == RETRIEVE_TOOLS_NAME:
            return await asyncio.to_thread(self._retrieve_tools, request)
        if tool_name_in_call == CALL_RETRIEVED_TOOL_NAME:
            native_request, error = self._native_tool_request(request)
            if error is not None:
                return error
            if native_request is None:
                return self._tool_message(
                    request,
                    {
                        "success": False,
                        "error": "failed to prepare native tool request",
                    },
                    status="error",
                )
            return await handler(native_request)
        return await handler(request)
