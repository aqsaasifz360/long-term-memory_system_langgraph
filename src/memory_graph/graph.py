from __future__ import annotations

import asyncio
import functools
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages
from langmem import create_memory_store_manager
from typing_extensions import Annotated, TypedDict
from langchain_core.runnables import RunnableConfig 

from memory_graph import configuration
from memory_graph.faiss_store import store_note_embedding, embeddings_model, FAISS_DIR

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class ProcessorState(State):
    function_name: str

logger = logging.getLogger("memory")

def manual_save_note_to_faiss(user_id: str, content: str, context: str = "") -> None:
    """Manually saves a notable memory into a FAISS vector store."""
    memory_to_store = {"content": content, "context": context}
    print(f"DEBUG: Manually calling store_note_embedding for user '{user_id}': {content[:50]}...")
    store_note_embedding(user_id, "Note", memory_to_store)

def get_store_manager(function_name: str, model: str, user_id: str, memory_types: list[configuration.MemoryConfig]):
    """Create store manager without caching to avoid unhashable type errors."""
    
    memory_config = next(conf for conf in memory_types if conf.name == function_name)

    kwargs: dict[str, Any] = {
        "enable_inserts": memory_config.update_mode in ["insert", "append"],
    }

    if memory_config.system_prompt:
        kwargs["instructions"] = memory_config.system_prompt

    print(f"DEBUG: Creating store manager for {function_name} with kwargs: {kwargs}")
    print(f"DEBUG: Memory config for {function_name}: update_mode={memory_config.update_mode}, parameters={memory_config.parameters}")

    return create_memory_store_manager(
        model,
        namespace=("memories", user_id, function_name),  # Use actual user_id instead of template
        **kwargs,
    )

@task()
async def process_memory_type(state: ProcessorState, config: RunnableConfig) -> None:
    """Processes messages to extract and store memories."""
    configurable = configuration.Configuration.from_context(config)
    user_id = configurable.user_id
    
    if not user_id or user_id == "default":
        print(f"DEBUG: Skipping memory processing for invalid user_id: {user_id}")
        return
    
    if not state["messages"] or len(state["messages"]) < 2:
        print(f"DEBUG: Not enough messages to extract memories for user {user_id}")
        return
    
    meaningful_messages = []
    for msg in state["messages"]:
        if hasattr(msg, 'content') and str(msg.content).strip():
            meaningful_messages.append(msg)
        elif hasattr(msg, 'type') and hasattr(msg, 'content'):
            content = str(msg.content).strip()
            if content:
                meaningful_messages.append(msg)
        else:
            content_str = str(msg).strip()
            if content_str and content_str != 'None':
                if hasattr(msg, 'type'):
                    meaningful_messages.append(msg)
                else:
                    meaningful_messages.append(HumanMessage(content=content_str))
    
    print(f"DEBUG: Filtered messages - Original: {len(state['messages'])}, Meaningful: {len(meaningful_messages)}")
    
    for i, msg in enumerate(state["messages"]):
        msg_type = getattr(msg, 'type', type(msg).__name__)
        msg_content = getattr(msg, 'content', str(msg))
        print(f"DEBUG: Original Message {i}: Type={msg_type}, Content='{str(msg_content)[:100]}'")
    
    for i, msg in enumerate(meaningful_messages):
        msg_type = getattr(msg, 'type', type(msg).__name__)
        msg_content = getattr(msg, 'content', str(msg))
        print(f"DEBUG: Meaningful Message {i}: Type={msg_type}, Content='{str(msg_content)[:100]}'")
    
    if len(meaningful_messages) < 1:
        print(f"DEBUG: No meaningful messages for memory extraction for user {user_id}")
        return

    print(f"DEBUG: Processing memory type: {state['function_name']} for user: {user_id}")
    print(f"DEBUG: Meaningful messages to process: {len(meaningful_messages)}")
    
    try:
        store_manager = get_store_manager(
            state["function_name"], 
            configurable.model, 
            user_id,
            configurable.memory_types
        )
        
        manager_input = {
            "messages": meaningful_messages, 
            "max_steps": configurable.max_extraction_steps
        }
        
        internal_llm_config = {
            "configurable": {
                "model": configurable.model, 
                "user_id": user_id
            }
        }
        
        print(f"DEBUG: Invoking store manager with input keys: {list(manager_input.keys())}")
        print(f"DEBUG: Internal LLM Config: {internal_llm_config}")
        
        try:
            manager_output = await store_manager.ainvoke(manager_input, config=internal_llm_config)
        except StopIteration as e:
            print(f"DEBUG: StopIteration caught for {state['function_name']} - likely no memories to extract")
            return
        except Exception as e:
            print(f"ERROR: Store manager invocation failed for {state['function_name']}: {e}")
            import traceback
            traceback.print_exc()
            return
        
        print(f"DEBUG: Manager output type: {type(manager_output)}")
        print(f"DEBUG: Manager output content: {manager_output}")
        
        # --- NEW LOGIC ADDED HERE ---
        if state["function_name"] == "Note" and isinstance(manager_output, list):
            print(f"DEBUG: Processing direct list output for Note memory type.")
            for item in manager_output:
                if isinstance(item, dict) and item.get('namespace') == ('memories', user_id, 'Note'):
                    extracted_content = item.get('value', {}).get('content', {})
                    if isinstance(extracted_content, dict) and 'content' in extracted_content:
                        note_content = extracted_content.get('content', '')
                        note_context = extracted_content.get('context', '') # Assuming context might be here too
                        if note_content:
                            print(f"DEBUG: Extracting direct note for FAISS: '{note_content[:100]}' with context '{note_context[:50]}'")
                            manual_save_note_to_faiss(user_id, note_content, note_context)
                        else:
                            print(f"WARNING: Extracted Note content is empty for user {user_id}.")
                    elif isinstance(extracted_content, str): # Handle cases where content is just a string
                        note_content = extracted_content
                        print(f"DEBUG: Extracting direct string note for FAISS: '{note_content[:100]}'")
                        manual_save_note_to_faiss(user_id, note_content, "") # No context for simple string
                    else:
                        print(f"WARNING: Unexpected content format for Note: {type(extracted_content)}")
            return # Processed list, no need to go to AIMessage section for Notes

        # --- EXISTING LOGIC FOR AIMessage (TOOL CALLS) ---
        if isinstance(manager_output, AIMessage):
            print(f"DEBUG: Processing AIMessage output for user {user_id}")
            print(f"DEBUG: Has tool_calls: {hasattr(manager_output, 'tool_calls') and bool(manager_output.tool_calls)}")
            
            if hasattr(manager_output, 'tool_calls') and manager_output.tool_calls:
                print(f"DEBUG: Found {len(manager_output.tool_calls)} tool calls")
                
                for i, tool_call in enumerate(manager_output.tool_calls):
                    print(f"DEBUG: Tool call {i}: {tool_call}")
                    
                    if state["function_name"] == "Note" and tool_call.get('name') in ['insert_document', 'update_document']:
                        args = tool_call.get('args', {})
                        namespace = args.get('namespace', [])
                        
                        if isinstance(namespace, list) and len(namespace) >= 2:
                            namespace_user_id = namespace[1] if len(namespace) > 1 else None
                            namespace_type = namespace[2] if len(namespace) > 2 else None
                            
                            if namespace_user_id == user_id and namespace_type == 'Note':
                                content_data = args.get('content', {})
                                
                                if isinstance(content_data, dict):
                                    note_content = content_data.get("content", "")
                                    note_context = content_data.get("context", "")
                                    
                                    if note_content:
                                        print(f"DEBUG: Storing note to FAISS (from tool call) for user {user_id}: {note_content[:100]}...")
                                        manual_save_note_to_faiss(user_id, note_content, note_context)
                                elif isinstance(content_data, str):
                                    print(f"DEBUG: Storing simple note to FAISS (from tool call) for user {user_id}: {content_data[:100]}...")
                                    manual_save_note_to_faiss(user_id, content_data, "")
                            else:
                                print(f"WARNING: Namespace user_id mismatch. Expected: {user_id}, Got: {namespace_user_id}")
            else:
                print(f"DEBUG: AIMessage has no tool calls or empty tool calls")
                print(f"DEBUG: AIMessage content: {getattr(manager_output, 'content', 'No content')}")
        else:
            print(f"DEBUG: Unexpected manager output type: {type(manager_output)}")
            
    except Exception as e:
        print(f"ERROR: Failed to process memory type {state['function_name']} for user {user_id}: {e}")
        import traceback
        traceback.print_exc()

@entrypoint(config_schema=configuration.Configuration)
async def graph(state: State, config: RunnableConfig) -> None:
    if not state["messages"]:
        print("DEBUG: No messages provided to memory graph")
        return

    print(f"DEBUG: Memory graph received {len(state['messages'])} messages")
    
    configurable = configuration.Configuration.from_context(config)
    print(f"DEBUG: Memory Configuration created for user_id: {configurable.user_id}")
    print(f"DEBUG: Processing {len(configurable.memory_types)} memory types")
    print(f"DEBUG: Model: {configurable.model}")
    print(f"DEBUG: User ID: {configurable.user_id}")

    if not configurable.user_id or configurable.user_id == "default":
        print(f"WARNING: Invalid or default user_id detected: {configurable.user_id}")
        return

    print(f"DEBUG: Processing {len(state['messages'])} total messages for memory extraction")
    
    for i, msg in enumerate(state["messages"]):
        msg_type = getattr(msg, 'type', type(msg).__name__)
        msg_content = getattr(msg, 'content', str(msg))
        print(f"DEBUG: Input Message {i}: Type={msg_type}, Content='{str(msg_content)[:100]}'")

    tasks = []
    for mem_type in configurable.memory_types:
        print(f"DEBUG: Creating task for memory type: {mem_type.name} for user {configurable.user_id}")
        task = process_memory_type(
            ProcessorState(messages=state["messages"], function_name=mem_type.name),
            config=config
        )
        tasks.append(task)
    
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        print(f"DEBUG: All memory processing tasks completed for user {configurable.user_id}")
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"ERROR: Task {i} failed for user {configurable.user_id} with: {result}")
                
    except Exception as e:
        print(f"ERROR: Failed to process memories for user {configurable.user_id}: {e}")
        import traceback
        traceback.print_exc()

__all__ = ["graph"]