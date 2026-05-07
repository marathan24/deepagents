"""Tests for OpenRouter compatibility middleware helpers."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from deepagents_cli.openrouter_compat import strip_openrouter_reasoning_messages


def test_strip_openrouter_reasoning_preserves_visible_message_data() -> None:
    """Provider-internal reasoning fragments should not replay across turns."""
    ai_message = AIMessage(
        content="I will inspect the file.",
        additional_kwargs={
            "reasoning_content": "internal",
            "reasoning_details": [{"type": "reasoning", "id": "rs_123"}],
            "tool_calls": [{"id": "call-1", "type": "function"}],
        },
        tool_calls=[
            {
                "id": "call-1",
                "name": "read_file",
                "args": {"file_path": "/tmp/file.txt"},
                "type": "tool_call",
            }
        ],
    )
    human_message = HumanMessage(content="continue")

    stripped = strip_openrouter_reasoning_messages([human_message, ai_message])

    assert stripped[0] is human_message
    stripped_ai = stripped[1]
    assert isinstance(stripped_ai, AIMessage)
    assert stripped_ai.content == "I will inspect the file."
    assert stripped_ai.tool_calls == ai_message.tool_calls
    assert stripped_ai.additional_kwargs == {
        "tool_calls": [{"id": "call-1", "type": "function"}]
    }
