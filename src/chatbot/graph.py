# chatbot/graph.py - Fixed version with proper user ID handling and memory validation

"""Example chatbot that incorporates user memories."""

import datetime
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_store
from langgraph.graph import StateGraph
from langgraph.graph.message import Messages, add_messages
from langgraph_sdk import get_client
from typing_extensions import Annotated

from chatbot.configuration import ChatConfigurable
from chatbot.utils import format_memories
from memory_graph.faiss_store import search_faiss
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

@dataclass
class ChatState:
    """The state of the chatbot."""
    messages: Annotated[list[Messages], add_messages]
    user_id: Optional[str] = None  # Add user_id to state

# Initialize the language model
llm = init_chat_model()

def deep_extract_content(data: Any) -> str:
    """Recursively extract content from nested data structures."""
    if isinstance(data, str):
        return data
    elif isinstance(data, dict):
        # Try common content fields in order of preference
        for key in ['content', 'text', 'data', 'description', 'message']:
            if key in data:
                extracted = deep_extract_content(data[key])
                if extracted.strip():
                    return extracted
        # If no content field found, try to extract from the whole dict
        return str(data)
    elif isinstance(data, list):
        # Extract from list items
        contents = []
        for item in data:
            extracted = deep_extract_content(item)
            if extracted.strip():
                contents.append(extracted)
        return '; '.join(contents)
    else:
        return str(data)

def format_memory_item(item) -> tuple[str, str]:
    """Extract content and memory type from a store item."""
    try:
        # Extract memory type from namespace
        memory_type = "Memory"
        if hasattr(item, 'namespace') and item.namespace:
            if len(item.namespace) >= 3:
                memory_type = item.namespace[2]  # Third element is memory type
        
        # Extract content from the nested structure
        content = ""
        if hasattr(item, 'value') and item.value:
            if isinstance(item.value, dict):
                # Handle the structure: {'kind': 'Memory', 'content': {'content': 'actual_data'}}
                if 'content' in item.value:
                    content = deep_extract_content(item.value['content'])
                else:
                    content = deep_extract_content(item.value)
            else:
                content = str(item.value)
        
        return memory_type, content.strip()
    except Exception as e:
        print(f"DEBUG: Error formatting memory item: {e}")
        return "Memory", str(item)

async def get_all_user_memories(user_id: str, query: str = "") -> Dict[str, List[str]]:
    """Retrieve all memories for a user, organized by type."""
    store = get_store()
    base_namespace = ("memories", user_id)
    
    memories_by_type = {}
    memory_types = ["User", "Note", "Action", "Procedural", "Episode"]
    
    for memory_type in memory_types:
        try:
            namespace = base_namespace + (memory_type,)
            print(f"DEBUG: Searching namespace: {namespace}")
            
            # Try both query search and list all
            items = []
            if query.strip():
                try:
                    items = await store.asearch(namespace, query=query, limit=20)
                    print(f"DEBUG: Query search returned {len(items) if items else 0} items for {memory_type}")
                except Exception as e:
                    print(f"DEBUG: Query search failed for {memory_type}: {e}")
            
            # If query search didn't return results, try listing all
            if not items:
                try:
                    items = await store.list(namespace, limit=50)
                    print(f"DEBUG: List all returned {len(items) if items else 0} items for {memory_type}")
                except Exception as e:
                    print(f"DEBUG: List all failed for {memory_type}: {e}")
            
            # Process and store the items
            if items:
                type_memories = []
                for item in items:
                    _, content = format_memory_item(item)
                    if content and content != "None":
                        type_memories.append(content)
                        print(f"DEBUG: Extracted {memory_type} memory: {content[:100]}...")
                
                if type_memories:
                    memories_by_type[memory_type] = type_memories
            
        except Exception as e:
            print(f"DEBUG: Error processing {memory_type} memories: {e}")
            import traceback
            traceback.print_exc()
    
    return memories_by_type

def determine_user_id(state: ChatState, config: RunnableConfig) -> str:
    """Determine the user ID from various sources - IMPROVED VERSION."""
    # Priority order:
    # 1. State user_id (if set)
    # 2. Config user_id
    # 3. Extract from messages that contain user ID patterns
    # 4. Default fallback
    
    if state.user_id:
        print(f"DEBUG: Using user_id from state: {state.user_id}")
        return state.user_id
    
    configurable = ChatConfigurable.from_context(config)
    if configurable.user_id != "default-user":
        print(f"DEBUG: Using user_id from config: {configurable.user_id}")
        return configurable.user_id
    
    # Try to extract user ID from any message in the conversation
    if state.messages:
        for message in state.messages:
            content = str(message.content) if hasattr(message, 'content') else str(message)
            
            # Look for explicit user ID patterns first
            import re
            user_id_patterns = [
                r"user[_\s]*id[:\s]*([A-Za-z0-9_]+)",
                r"id[:\s]*([A-Za-z0-9_]+)",
                r"User_(\d+)",  # Specific pattern for User_XXXX format
                r"user_(\d+)",  # Alternative user_XXXX format
            ]
            
            for pattern in user_id_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    potential_user_id = match.group(1) if len(match.groups()) == 1 else match.group(0)
                    print(f"DEBUG: Extracted user ID from message: {potential_user_id}")
                    return potential_user_id
            
            # Fallback to name patterns if no explicit ID found
            name_patterns = [
                r"my name is (\w+)",
                r"i'?m (\w+)",
                r"this is (\w+)",
                r"call me (\w+)",
                r"it'?s (\w+)",  # Added pattern for "it's Mona"
                r"hello,?\s+(\w+)",  # Added pattern for "Hello Mona"
            ]
            
            for pattern in name_patterns:
                match = re.search(pattern, content.lower())
                if match:
                    potential_user_id = match.group(1)
                    print(f"DEBUG: Extracted potential user ID from name: {potential_user_id}")
                    return potential_user_id
    
    print("DEBUG: Using default user ID")
    return "default-user"

async def handle_user_identification(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    """Handle user identification and return updated state."""
    user_id = determine_user_id(state, config)
    
    if user_id != "default-user":
        print(f"DEBUG: User identified as: {user_id}")
        # Update the state with the identified user
        new_state = {"user_id": user_id}
        
        # If this is a new user identification, add a welcome message
        if not state.user_id or state.user_id != user_id:
            from langchain_core.messages import AIMessage
            welcome_msg = AIMessage(
                content=f"Hello {user_id}! Nice to meet you. How can I help you today?"
            )
            new_state["messages"] = [welcome_msg]
        
        return new_state
    
    return {}

async def bot(state: ChatState, config: RunnableConfig) -> dict[str, list[Messages]]:
    """The core chatbot logic: responds to user and incorporates memory."""
    # Determine the user ID for this conversation
    user_id = determine_user_id(state, config)
    
    # Update config with the determined user ID
    updated_config = dict(config)
    updated_config["configurable"] = dict(config.get("configurable", {}))
    updated_config["configurable"]["user_id"] = user_id
    
    configurable = ChatConfigurable.from_context(updated_config)
    
    print(f"DEBUG: Bot processing for user: {user_id}")

    # Get the latest user message for query context
    latest_message = state.messages[-1] if state.messages else ""
    query = str(latest_message.content) if hasattr(latest_message, 'content') else str(latest_message)
    
    print(f"DEBUG: Processing query for user '{user_id}': {query[:100]}...")

    # Get all stored memories - ENSURE we're using the correct user_id
    all_memories = await get_all_user_memories(user_id, query)
    
    # Search FAISS for episodic memories - ENSURE we're using the correct user_id
    faiss_results = []
    try:
        faiss_results = search_faiss(user_id, "Note", query, k=5)
        print(f"DEBUG: FAISS search returned {len(faiss_results)} results for user {user_id}")
    except Exception as e:
        print(f"DEBUG: FAISS search failed for user {user_id}: {e}")

    # Build comprehensive memory section
    memory_parts = []
    
    # Add stored memories by type
    for memory_type, memories in all_memories.items():
        if memories:
            memory_parts.append(f"**{memory_type} Information:**")
            for memory in memories:
                memory_parts.append(f"- {memory}")
            memory_parts.append("")  # Add spacing
    
    # Add FAISS memories if available
    if faiss_results:
        memory_parts.append("**Recent Notes from Past Conversations:**")
        for doc in faiss_results:
            content = doc.page_content
            context = doc.metadata.get('context', '')
            if context:
                memory_parts.append(f"- {content} (Context: {context})")
            else:
                memory_parts.append(f"- {content}")
        memory_parts.append("")

    # Create the final memory section - FIXED: Only include if memories exist
    memory_section = ""
    if memory_parts:
        memory_content = "\n".join(memory_parts).strip()
        memory_section = f"\n\n## Your Memory About {user_id}\n\n{memory_content}"
        print(f"DEBUG: Created memory section with {len(memory_content)} characters for user {user_id}")
        print(f"DEBUG: Memory section preview:\n{memory_section[:500]}...")
    else:
        print(f"DEBUG: No memories found for user: {user_id}")
        # For new users, explicitly state no previous memories
        if user_id != "default-user":
            memory_section = f"\n\n## Your Memory About {user_id}\n\nThis appears to be your first conversation with {user_id}. You have no previous memories about them yet."
    
    # Compose the system prompt
    prompt = configurable.system_prompt.format(
        user_info=memory_section,
        time=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    )
    
    print(f"DEBUG: Final system prompt length: {len(prompt)}")

    # Prepare messages for LLM
    try:
        messages_for_llm = [{"role": "system", "content": prompt}]
        
        # Convert state messages to proper format
        for msg in state.messages:
            if hasattr(msg, 'type') and hasattr(msg, 'content'):
                role = "assistant" if msg.type == "ai" else "user"
                messages_for_llm.append({"role": role, "content": msg.content})
            else:
                # Fallback for other message types
                messages_for_llm.append({"role": "user", "content": str(msg)})
        
        print(f"DEBUG: Prepared {len(messages_for_llm)} messages for LLM")
        
        # Invoke the LLM with updated config
        response = await llm.ainvoke(
            messages_for_llm,
            config={"configurable": {"model": configurable.model}},
        )
        
        print(f"DEBUG: LLM response generated successfully for user {user_id}")
        
        # Return response with updated user_id in state
        return {
            "messages": [response],
            "user_id": user_id
        }
        
    except Exception as e:
        print(f"ERROR: Failed to get LLM response for user {user_id}: {e}")
        import traceback
        traceback.print_exc()
        
        # Return a fallback response
        from langchain_core.messages import AIMessage
        fallback = AIMessage(content="I'm having trouble accessing my memories right now. How can I help you?")
        return {
            "messages": [fallback],
            "user_id": user_id
        }

async def schedule_memories(state: ChatState, config: RunnableConfig) -> None:
    """Schedule a memory extraction run to debounce memory formation."""
    # Use the user_id from state if available, otherwise determine it
    user_id = state.user_id or determine_user_id(state, config)
    
    # Skip memory scheduling for default users or if no real conversation
    if user_id == "default-user" or not state.messages:
        print(f"DEBUG: Skipping memory extraction for user: {user_id}")
        return
    
    # Update config with the correct user ID
    updated_config = dict(config)
    updated_config["configurable"] = dict(config.get("configurable", {}))
    updated_config["configurable"]["user_id"] = user_id
    
    configurable = ChatConfigurable.from_context(updated_config)
    
    print(f"DEBUG: Scheduling memory extraction for user: {user_id}")
    print(f"DEBUG: Processing {len(state.messages)} messages")
    
    # Print the actual messages being processed
    for i, msg in enumerate(state.messages):
        msg_preview = str(msg.content)[:100] if hasattr(msg, 'content') else str(msg)[:100]
        print(f"DEBUG: Message {i}: {type(msg).__name__} - {msg_preview}")

    try:
        memory_client = get_client()

        await memory_client.runs.create(
            thread_id=config["configurable"]["thread_id"],
            multitask_strategy="enqueue",
            after_seconds=configurable.delay_seconds,
            assistant_id=configurable.mem_assistant_id,
            input={"messages": state.messages},
            config={
                "configurable": {
                    "user_id": user_id,
                    "memory_types": configurable.memory_types,
                }
            },
        )
        print(f"DEBUG: Memory extraction scheduled successfully for user {user_id}")
    except Exception as e:
        print(f"ERROR: Failed to schedule memory extraction for user {user_id}: {e}")
        import traceback
        traceback.print_exc()

# Debugging function to manually check memories
async def debug_user_memories(user_id: str):
    """Debug function to inspect all memories for a user."""
    print(f"\n=== DEBUGGING MEMORIES FOR USER: {user_id} ===")
    
    memories = await get_all_user_memories(user_id)
    
    if not memories:
        print("No memories found!")
        return
    
    for memory_type, contents in memories.items():
        print(f"\n{memory_type} ({len(contents)} items):")
        for i, content in enumerate(contents):
            print(f"  {i+1}. {content}")
    
    print("=== END MEMORY DEBUG ===\n")

# Build the LangGraph agent
builder = StateGraph(ChatState, config_schema=ChatConfigurable)
builder.add_node("identify_user", handle_user_identification)
builder.add_node("bot", bot)
builder.add_node("schedule_memories", schedule_memories)

# Updated flow to handle user identification first
builder.add_edge("__start__", "identify_user")
builder.add_edge("identify_user", "bot")
builder.add_edge("bot", "schedule_memories")

graph = builder.compile()