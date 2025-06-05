# chatbot/configuration.py - Updated version

"""Define the configurable parameters for the chatbot."""

import os
from dataclasses import dataclass, fields
from typing import Any, Optional
from langgraph.config import get_config
from langchain_core.runnables import RunnableConfig
from chatbot.prompts import SYSTEM_PROMPT

@dataclass(kw_only=True)
class ChatConfigurable:
    """The configurable fields for the chatbot."""

    user_id: str = "default-user"  # This will be overridden by user input
    mem_assistant_id: str = "memory_graph"  
    model: str = "gemini-2.0-flash"
    delay_seconds: int = 3  # For debouncing memory creation
    system_prompt: str = SYSTEM_PROMPT
    memory_types: Optional[list[dict]] = None
    """The memory types for the memory assistant."""

    @classmethod
    def from_context(cls, config: Optional[RunnableConfig] = None) -> "ChatConfigurable":
        """Create a ChatConfigurable instance from a RunnableConfig object or environment variables."""
        configurable = config.get("configurable", {}) if config else {}

        values: dict[str, Any] = {}
        for f in fields(cls):
            if f.init:
                value = configurable.get(f.name)
                if value is None:
                    value = os.environ.get(f.name.upper())
                if value is not None:
                    values[f.name] = value

        return cls(**{k: v for k, v in values.items() if v is not None})

    @classmethod
    def create_for_user(cls, user_id: str, **kwargs) -> "ChatConfigurable":
        """Create a ChatConfigurable instance for a specific user."""
        return cls(user_id=user_id, **kwargs)