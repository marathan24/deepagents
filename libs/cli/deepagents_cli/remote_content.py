"""Remote graph content-block compatibility helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.types import Command

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langgraph.prebuilt.tool_node import ToolCallRequest


def _image_data_url(block: dict[str, Any]) -> str | None:
    """Return a data URL for a legacy image block when it is well-formed."""
    encoded = block.get("base64")
    if not isinstance(encoded, str) or not encoded:
        return None
    mime_type = block.get("mime_type")
    if not isinstance(mime_type, str) or not mime_type:
        mime_type = "image/png"
    return f"data:{mime_type};base64,{encoded}"


def normalize_remote_content(content: Any) -> Any:  # noqa: ANN401
    """Convert content blocks unsupported by LangGraph remote into accepted forms.

    Returns:
        The original content, or a normalized copy when legacy blocks are found.
    """
    if not isinstance(content, list):
        return content

    changed = False
    normalized: list[Any] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "image":
            data_url = _image_data_url(block)
            if data_url is not None:
                normalized.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    }
                )
                changed = True
                continue
        normalized.append(block)

    return normalized if changed else content


def normalize_tool_message_for_remote(message: ToolMessage) -> ToolMessage:
    """Return `message` with remote-compatible content blocks."""
    normalized = normalize_remote_content(message.content)
    if normalized is message.content:
        return message
    return message.model_copy(update={"content": normalized})


def normalize_tool_result_for_remote(
    result: ToolMessage | Command[Any],
) -> ToolMessage | Command[Any]:
    """Normalize any `ToolMessage` content nested in a tool result.

    Returns:
        The original tool result, or a copy with compatible content blocks.
    """
    if isinstance(result, ToolMessage):
        return normalize_tool_message_for_remote(result)

    if not isinstance(result, Command) or not isinstance(result.update, dict):
        return result

    messages = result.update.get("messages")
    if not isinstance(messages, list):
        return result

    changed = False
    normalized_messages: list[Any] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            normalized_message = normalize_tool_message_for_remote(message)
            changed = changed or normalized_message is not message
            normalized_messages.append(normalized_message)
        else:
            normalized_messages.append(message)

    if not changed:
        return result

    return Command(
        graph=result.graph,
        update={**result.update, "messages": normalized_messages},
        resume=result.resume,
        goto=result.goto,
    )


class RemoteContentCompatibilityMiddleware(AgentMiddleware):
    """Normalize tool output content so it can cross LangGraph remote streams."""

    def wrap_tool_call(  # noqa: PLR6301  # AgentMiddleware hook
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Normalize the downstream tool result.

        Returns:
            A remote-compatible tool result.
        """
        return normalize_tool_result_for_remote(handler(request))

    async def awrap_tool_call(  # noqa: PLR6301  # AgentMiddleware hook
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Async variant of `wrap_tool_call`.

        Returns:
            A remote-compatible tool result.
        """
        return normalize_tool_result_for_remote(await handler(request))
