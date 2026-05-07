"""Tests for remote content block compatibility."""

from __future__ import annotations

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from deepagents_cli.remote_content import normalize_tool_result_for_remote
from deepagents_cli.tool_display import format_tool_message_content


def test_normalizes_legacy_image_tool_message_to_image_url() -> None:
    """Legacy SDK image blocks should serialize through LangGraph remote."""
    message = ToolMessage(
        content_blocks=[
            {
                "type": "image",
                "base64": "YWJj",
                "mime_type": "image/png",
            }
        ],
        name="read_file",
        tool_call_id="tool-1",
    )

    result = normalize_tool_result_for_remote(message)

    assert isinstance(result, ToolMessage)
    assert result.content == [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,YWJj"},
        }
    ]
    assert result.name == "read_file"
    assert result.tool_call_id == "tool-1"


def test_normalizes_legacy_image_in_command_messages() -> None:
    """Command updates can also contain ToolMessages that need normalization."""
    message = ToolMessage(
        content=[
            {
                "type": "image",
                "base64": "YWJj",
                "mime_type": "image/jpeg",
            }
        ],
        tool_call_id="tool-1",
    )
    command = Command(update={"messages": [message], "other": "value"})

    result = normalize_tool_result_for_remote(command)

    assert isinstance(result, Command)
    assert result.update["other"] == "value"
    normalized_message = result.update["messages"][0]
    assert isinstance(normalized_message, ToolMessage)
    assert normalized_message.content == [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,YWJj"},
        }
    ]


def test_display_redacts_data_image_url() -> None:
    """Data image URLs should not dump base64 content into CLI output."""
    content = [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,YWJj"},
        }
    ]

    assert format_tool_message_content(content) == "[Image: image/png, ~0KB]"
