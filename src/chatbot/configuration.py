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
    
    # Enhanced debouncing configuration
    delay_seconds: int = 30  # Default inactivity threshold before memory extraction
    min_messages_for_memory: int = 2  # Minimum messages before considering memory extraction
    max_delay_seconds: int = 300  # Maximum delay (5 minutes) before forcing memory extraction
    
    system_prompt: str = SYSTEM_PROMPT
    memory_types: Optional[list[dict]] = None
    """The memory types for the memory assistant."""
    
    # Advanced memory settings
    enable_memory_debouncing: bool = True  # Enable/disable debouncing
    force_memory_on_context_switch: bool = True  # Force memory save when user switches topics
    memory_batch_size: int = 10  # Number of messages to batch for memory extraction

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
                    # Handle type conversion for numeric fields
                    if f.type in [int, Optional[int]] and isinstance(value, str):
                        try:
                            value = int(value)
                        except ValueError:
                            print(f"WARNING: Could not convert {f.name} value '{value}' to int, using default")
                            continue
                    elif f.type in [bool, Optional[bool]] and isinstance(value, str):
                        value = value.lower() in ['true', '1', 'yes', 'on']
                    
                    values[f.name] = value

        return cls(**{k: v for k, v in values.items() if v is not None})

    @classmethod
    def create_for_user(cls, user_id: str, **kwargs) -> "ChatConfigurable":
        """Create a ChatConfigurable instance for a specific user."""
        return cls(user_id=user_id, **kwargs)
    
    def get_effective_delay(self, message_count: int = 0) -> int:
        """Calculate effective delay based on message count and configuration."""
        if not self.enable_memory_debouncing:
            return 0  # No delay if debouncing disabled
        
        # Scale delay based on message volume (more messages = shorter delay)
        if message_count > 10:
            return max(self.delay_seconds // 2, 5)  # Reduce delay for active conversations
        elif message_count > 5:
            return max(int(self.delay_seconds * 0.75), 10)
        else:
            return self.delay_seconds
    
    def should_force_memory_extraction(self, time_since_last_extraction: float) -> bool:
        """Determine if memory extraction should be forced regardless of activity."""
        return time_since_last_extraction >= self.max_delay_seconds