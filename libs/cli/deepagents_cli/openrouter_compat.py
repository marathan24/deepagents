"""OpenRouter compatibility middleware for multi-turn agent loops."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, BaseMessage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from langchain.agents.middleware.types import ModelRequest, ModelResponse


logger = logging.getLogger(__name__)

_OPENROUTER_REASONING_KEYS = frozenset({"reasoning_content", "reasoning_details"})


def _is_openrouter_model(model: object) -> bool:
    """Return whether `model` is an OpenRouter chat model."""
    try:
        ls_params = model._get_ls_params()  # type: ignore[attr-defined]
    except (AttributeError, TypeError, RuntimeError):
        module = type(model).__module__
        return module.startswith("langchain_openrouter")
    return isinstance(ls_params, dict) and ls_params.get("ls_provider") == "openrouter"


def strip_openrouter_reasoning_messages(
    messages: Sequence[BaseMessage],
) -> list[BaseMessage]:
    """Remove OpenRouter reasoning replay metadata from assistant messages.

    Returns:
        A message list suitable for a follow-up OpenRouter tool-calling turn.
    """
    stripped: list[BaseMessage] = []
    changed = False
    for message in messages:
        if isinstance(message, AIMessage):
            reasoning_keys = (
                message.additional_kwargs.keys() & _OPENROUTER_REASONING_KEYS
            )
            if reasoning_keys:
                additional_kwargs = {
                    key: value
                    for key, value in message.additional_kwargs.items()
                    if key not in _OPENROUTER_REASONING_KEYS
                }
                stripped.append(
                    message.model_copy(
                        update={"additional_kwargs": additional_kwargs}
                    )
                )
                changed = True
                continue
        stripped.append(message)

    return stripped if changed else list(messages)


def _strip_request_reasoning(request: ModelRequest[Any]) -> ModelRequest[Any]:
    """Strip OpenRouter reasoning fragments from a model request when needed.

    Returns:
        The original request, or a request copy with sanitized messages.
    """
    if not _is_openrouter_model(request.model):
        return request

    messages = strip_openrouter_reasoning_messages(request.messages)
    if messages == request.messages:
        return request

    logger.debug("Stripped OpenRouter reasoning metadata before model call")
    return request.override(messages=messages)


class OpenRouterReasoningCompatibilityMiddleware(AgentMiddleware):
    """Prevent unsupported OpenRouter reasoning item replay in tool loops."""

    def wrap_model_call(  # noqa: PLR6301  # AgentMiddleware hook
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse],
    ) -> ModelResponse:
        """Strip OpenRouter-only reasoning metadata and delegate.

        Returns:
            The downstream model response.
        """
        return handler(_strip_request_reasoning(request))

    async def awrap_model_call(  # noqa: PLR6301  # AgentMiddleware hook
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async variant of `wrap_model_call`.

        Returns:
            The downstream model response.
        """
        return await handler(_strip_request_reasoning(request))
