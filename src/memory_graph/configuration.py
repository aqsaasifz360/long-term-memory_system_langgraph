# src/memory_graph/configuration.py - Updated version

"""Define the configurable parameters for the memory service."""

import os
from dataclasses import dataclass, field, fields
from typing import Any, Literal, Optional
from langgraph.config import get_config
from langchain_core.runnables import RunnableConfig
from typing_extensions import Annotated

@dataclass(kw_only=True)
class MemoryConfig:
    """Configuration for memory-related operations."""

    name: str
    """This tells the model how to reference the function and organizes related memories within the namespace."""

    description: str
    """Description for what this memory type is intended to capture."""

    parameters: dict[str, Any]
    """The JSON Schema of the memory document to manage."""

    system_prompt: str = ""
    """The system prompt to use for the memory assistant."""

    update_mode: Literal["patch", "insert", "append"] = field(default="patch")
    """Whether to continuously patch the memory, or treat each new generation as a new memory.

    - `patch`: Useful for a structured profile or updatable memory.
    - `insert`: Save everything as a new entry.
    - `append`: Similar to insert, with emphasis on accumulation.
    """

@dataclass(kw_only=True)
class Configuration:
    """Main configuration class for the memory graph system."""

    user_id: str = "default"
    """The ID of the user to remember in the conversation."""

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="gemini-2.0-flash",
        metadata={
            "description": "The name of the language model to use for the agent. "
                           "Should be in the form: provider/model-name."
        },
    )
    """The model to use for generating memories."""

    memory_types: list[MemoryConfig] = field(default_factory=list)
    """The memory_types for the memory assistant."""

    max_extraction_steps: int = 1
    """The maximum number of steps to take when extracting memories."""

    @classmethod
    def from_context(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig or environment."""
        configurable = config.get("configurable", {}) if config else {}

        values: dict[str, Any] = {}
        for f in fields(cls):
            if f.init:
                value = configurable.get(f.name)
                if value is None:
                    value = os.environ.get(f.name.upper())
                if value is not None:
                    values[f.name] = value

        # Handle memory_types configuration
        if values.get("memory_types") is None:
            # If memory_types are not provided, use DEFAULT_MEMORY_CONFIGS directly
            values["memory_types"] = DEFAULT_MEMORY_CONFIGS
        else:
            # If memory_types are provided, ensure they are MemoryConfig instances
            values["memory_types"] = [
                MemoryConfig(**v) if isinstance(v, dict) else v
                for v in (values["memory_types"] or [])
            ]

        return cls(**{k: v for k, v in values.items() if v is not None})

    @classmethod
    def create_for_user(cls, user_id: str, **kwargs) -> "Configuration":
        """Create a Configuration instance for a specific user."""
        return cls(user_id=user_id, **kwargs)

DEFAULT_MEMORY_CONFIGS = [
    MemoryConfig(
        name="User",
        description="Update this document to maintain up-to-date information about the user in the conversation.",
        update_mode="patch",
        system_prompt=(
            "You are an assistant whose sole purpose is to extract and update the user's profile information "
            "from the conversation. When the user shares details about themselves such as their name, age, "
            "interests, home, occupation, or conversation preferences, call the `update_document` tool to store "
            "this information in the user's profile. Prioritize extracting concrete details. Only update fields "
            "for which you have clear, specific information."
        ),
        parameters={
            "type": "object",
            "properties": {
                "user_name": {"type": "string", "description": "The user's preferred name"},
                "age": {"type": "integer", "description": "The user's age"},
                "interests": {
                    "type": "array", "items": {"type": "string"},
                    "description": "A list of the user's interests"
                },
                "home": {"type": "string", "description": "The user's hometown or neighborhood"},
                "occupation": {"type": "string", "description": "The user's current occupation or profession"},
                "conversation_preferences": {
                    "type": "array", "items": {"type": "string"},
                    "description": "User's preferred conversation styles, pronouns, or off-limit topics"
                },
            },
            "required": []
        },
    ),
    MemoryConfig(
        name="Note",
        description=(
            "Use this tool to record any information that is relevant to the user, but that is not directly related "
            "to their identity or preferences. This could include facts they state, past events, or interests."
        ),
        update_mode="append",
        parameters={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "A concise summary of a piece of information relevant to the user."
                }
            },
            "required": ["content"]
        },
    ),
    MemoryConfig(
        name="Action",
        description=(
            "Use this tool to record any actions the user wants you to take or goals they express. "
            "This includes reminders, tasks, or requests."
        ),
        update_mode="append",
        parameters={
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "A clear and actionable description of the user's requested action or goal."
                },
                "due_date": {
                    "type": "string",
                    "format": "date",
                    "description": "Optional due date (YYYY-MM-DD) for the action."
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Optional priority level of the action."
                }
            },
            "required": ["description"]
        },
    ),
    MemoryConfig(
        name="Procedural",
        description=(
            "Use this tool to record knowledge about how to perform specific tasks or procedures relevant to the "
            "user's interactions or preferences. Includes step-by-step instructions or workflows."
        ),
        update_mode="append",
        parameters={
            "type": "object",
            "properties": {
                "procedure_name": {
                    "type": "string",
                    "description": "A concise name for the procedure or task."
                },
                "steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ordered steps to perform the procedure."
                },
                "context": {
                    "type": "string",
                    "description": "Optional context or conditions under which this procedure applies."
                }
            },
            "required": ["procedure_name", "steps"]
        },
    )
]