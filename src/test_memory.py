"""Test script to validate the memory system functionality."""

import asyncio
import uuid
from chatbot.graph import graph


async def test_memory_persistence():
    """Test that memories persist across different conversation threads."""
    
    user_id = f"test_user_{uuid.uuid4().hex[:8]}"
    
    # First conversation - user shares personal info
    print("=== First Conversation ===")
    thread_id_1 = f"thread_{uuid.uuid4().hex[:8]}"
    
    config_1 = {
        "configurable": {
            "user_id": user_id,
            "thread_id": thread_id_1,
        }
    }
    
    messages_1 = [
        {"role": "user", "content": "Hi! My name is Alice and I'm a software engineer from San Francisco. I love hiking and reading sci-fi novels."}
    ]
    
    result_1 = await graph.ainvoke({"messages": messages_1}, config=config_1)
    print("Bot:", result_1["messages"][-1].content)
    
    # Add a delay to allow memory processing
    await asyncio.sleep(5)
    
    # Second conversation - new thread, same user
    print("\n=== Second Conversation (New Thread) ===")
    thread_id_2 = f"thread_{uuid.uuid4().hex[:8]}"
    
    config_2 = {
        "configurable": {
            "user_id": user_id,
            "thread_id": thread_id_2,
        }
    }
    
    messages_2 = [
        {"role": "user", "content": "Hello again! How are you today?"}
    ]
    
    result_2 = await graph.ainvoke({"messages": messages_2}, config=config_2)
    print("Bot:", result_2["messages"][-1].content)
    
    # Third conversation - user updates info
    print("\n=== Third Conversation (Updated Info) ===")
    messages_3 = [
        {"role": "user", "content": "Actually, I just moved to Portland and started working as a data scientist!"}
    ]
    
    result_3 = await graph.ainvoke({"messages": messages_3}, config=config_2)
    print("Bot:", result_3["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(test_memory_persistence())
