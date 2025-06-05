import asyncio
import os
from langgraph_sdk import get_client
from langgraph.config import get_config
from chatbot.configuration import ChatConfigurable # Assuming chatbot.configuration is accessible

# Load environment variables (ensure .env is in the correct path or variables are system-wide)
from dotenv import load_dotenv
load_dotenv(dotenv_path="D:\\Zikra LLC\\customer-success-draft\\.env.example") # Adjust path if needed

# --- Configuration for testing ---
# Make sure these match how your LangGraph server is configured
# If using local dev, these might be set by langgraph.cli run --app ...
LANGGRAPH_API_URL = os.getenv("LANGGRAPH_API_URL", "http://localhost:8000")
LANGGRAPH_API_KEY = os.getenv("LANGGRAPH_API_KEY")

# Create a client to interact with your deployed LangGraph agent
client = get_client(api_url=LANGGRAPH_API_URL, api_key=LANGGRAPH_API_KEY)

# --- IMPORTANT: Set your actual chatbot's assistant ID here ---
# This is the name/ID you used when deploying your 'chatbot/graph.py'
# For example, if you deployed it using 'langgraph deploy --app chatbot.graph:graph --name my-super-bot'
# then MAIN_CHATBOT_ASSISTANT_ID would be 'my-super-bot'.
# If you're running locally with `langgraph run`, the default might be the module name.
MAIN_CHATBOT_ASSISTANT_ID = "your-main-chatbot-app-name" # <<< --- REPLACE THIS PLACEHOLDER!

# MEMORY_ASSISTANT_ID can be derived from ChatConfigurable's default
# This assumes your memory graph is deployed under the same ID as configured in chatbot/configuration.py
MEMORY_ASSISTANT_ID = ChatConfigurable.from_context().mem_assistant_id


async def test_user_memory_recall():
    print(f"--- Testing User Memory Recall ---")
    print(f"Using MAIN_CHATBOT_ASSISTANT_ID: {MAIN_CHATBOT_ASSISTANT_ID}")
    print(f"Using MEMORY_ASSISTANT_ID: {MEMORY_ASSISTANT_ID}")


    # --- User 1: Alice ---
    user_alice_id = "test_user_alice_123"
    print(f"\n--- Scenario: Alice (ID: {user_alice_id}) ---")

    # === Thread 1 for Alice: Learning Information ===
    print("\n--- Alice - Thread 1: Learning Facts ---")
    thread_alice_1 = client.threads.create()
    print(f"  New Thread ID for Alice (1): {thread_alice_1.id}")

    # Interaction 1: Name and age
    print("  Alice says: 'Hi, my name is Alice, and I am 28 years old.'")
    await client.threads.invoke(
        thread_id=thread_alice_1.id,
        assistant_id=MAIN_CHATBOT_ASSISTANT_ID,
        input={"messages": ("human", "Hi, my name is Alice, and I am 28 years old.")},
        config={"configurable": {"user_id": user_alice_id}}
    )
    # Wait for memory to be processed (adjust delay_seconds if you changed it)
    await asyncio.sleep(ChatConfigurable.from_context().delay_seconds + 2)
    print("  (Memory processing scheduled and likely completed for Alice's name and age)")

    # Interaction 2: Job and interests
    print("  Alice says: 'I work as a data scientist, and I love hiking and reading sci-fi novels.'")
    await client.threads.invoke(
        thread_id=thread_alice_1.id,
        assistant_id=MAIN_CHATBOT_ASSISTANT_ID,
        input={"messages": ("human", "I work as a data scientist, and I love hiking and reading sci-fi novels.")},
        config={"configurable": {"user_id": user_alice_id}}
    )
    await asyncio.sleep(ChatConfigurable.from_context().delay_seconds + 2)
    print("  (Memory processing scheduled and likely completed for Alice's job and interests)")

    # Interaction 3: Preference (Note type memory)
    print("  Alice says: 'By the way, I prefer using metric units.'")
    response_stream = client.threads.invoke(
        thread_id=thread_alice_1.id,
        assistant_id=MAIN_CHATBOT_ASSISTANT_ID,
        input={"messages": ("human", "By the way, I prefer using metric units.")},
        config={"configurable": {"user_id": user_alice_id}}
    )
    async for chunk in response_stream:
        if "messages" in chunk["output"] and chunk["output"]["messages"]:
            print(f"  Bot replies (Thread 1): {chunk['output']['messages'][-1].content}")
    await asyncio.sleep(ChatConfigurable.from_context().delay_seconds + 2)
    print("  (Memory processing scheduled and likely completed for Alice's preference)")


    # === Thread 2 for Alice: Recalling Information (NEW THREAD) ===
    print("\n--- Alice - Thread 2: Recalling Facts (NEW THREAD) ---")
    thread_alice_2 = client.threads.create() # Create a brand new thread for Alice
    print(f"  New Thread ID for Alice (2): {thread_alice_2.id}")

    # Recall question 1: Name
    print("  Alice says: 'What's my name?'")
    response_stream = client.threads.invoke(
        thread_id=thread_alice_2.id,
        assistant_id=MAIN_CHATBOT_ASSISTANT_ID,
        input={"messages": ("human", "What's my name?")},
        config={"configurable": {"user_id": user_alice_id}} # IMPORTANT: Use the same user_id
    )
    async for chunk in response_stream:
        if "messages" in chunk["output"] and chunk["output"]["messages"]:
            print(f"  Bot replies (Thread 2): {chunk['output']['messages'][-1].content}")
            assert "alice" in chunk['output']['messages'][-1].content.lower()
            print("  Assertion Passed: Bot recalled Alice's name.")

    # Recall question 2: Job and interests
    print("  Alice says: 'What do you remember about my job and hobbies?'")
    response_stream = client.threads.invoke(
        thread_id=thread_alice_2.id,
        assistant_id=MAIN_CHATBOT_ASSISTANT_ID,
        input={"messages": ("human", "What do you remember about my job and hobbies?")},
        config={"configurable": {"user_id": user_alice_id}}
    )
    async for chunk in response_stream:
        if "messages" in chunk["output"] and chunk["output"]["messages"]:
            print(f"  Bot replies (Thread 2): {chunk['output']['messages'][-1].content}")
            assert "data scientist" in chunk['output']['messages'][-1].content.lower()
            assert ("hiking" in chunk['output']['messages'][-1].content.lower() or "reading sci-fi" in chunk['output']['messages'][-1].content.lower())
            print("  Assertion Passed: Bot recalled Alice's job and hobbies.")

    # Recall question 3: Preference (Note type memory)
    print("  Alice says: 'Did I tell you about my unit preferences?'")
    response_stream = client.threads.invoke(
        thread_id=thread_alice_2.id,
        assistant_id=MAIN_CHATBOT_ASSISTANT_ID,
        input={"messages": ("human", "Did I tell you about my unit preferences?")},
        config={"configurable": {"user_id": user_alice_id}}
    )
    async for chunk in response_stream:
        if "messages" in chunk["output"] and chunk["output"]["messages"]:
            print(f"  Bot replies (Thread 2): {chunk['output']['messages'][-1].content}")
            assert "metric" in chunk['output']['messages'][-1].content.lower()
            print("  Assertion Passed: Bot recalled Alice's unit preference.")


    # --- User 2: Bob (Separate User, should have no knowledge of Alice) ---
    user_bob_id = "test_user_bob_456"
    print(f"\n--- Scenario: Bob (ID: {user_bob_id}) ---")

    # === Thread 1 for Bob: Learning Information ===
    print("\n--- Bob - Thread 1: Learning Facts ---")
    thread_bob_1 = client.threads.create()
    print(f"  New Thread ID for Bob (1): {thread_bob_1.id}")

    print("  Bob says: 'My name is Bob. I am 45 years old and I'm a chef.'")
    response_stream = client.threads.invoke(
        thread_id=thread_bob_1.id,
        assistant_id=MAIN_CHATBOT_ASSISTANT_ID,
        input={"messages": ("human", "My name is Bob. I am 45 years old and I'm a chef.")},
        config={"configurable": {"user_id": user_bob_id}}
    )
    async for chunk in response_stream:
        if "messages" in chunk["output"] and chunk["output"]["messages"]:
            print(f"  Bot replies (Thread 1): {chunk['output']['messages'][-1].content}")
    await asyncio.sleep(ChatConfigurable.from_context().delay_seconds + 2)
    print("  (Memory processing scheduled and likely completed for Bob)")

    # === Thread 2 for Bob: Recalling Information (NEW THREAD) ===
    print("\n--- Bob - Thread 2: Recalling Facts (NEW THREAD) ---")
    thread_bob_2 = client.threads.create() # Create a brand new thread for Bob
    print(f"  New Thread ID for Bob (2): {thread_bob_2.id}")

    # Recall question for Bob: Name and job
    print("  Bob says: 'Who am I and what do I do?'")
    response_stream = client.threads.invoke(
        thread_id=thread_bob_2.id,
        assistant_id=MAIN_CHATBOT_ASSISTANT_ID,
        input={"messages": ("human", "Who am I and what do I do?")},
        config={"configurable": {"user_id": user_bob_id}} # IMPORTANT: Use the same user_id for Bob
    )
    async for chunk in response_stream:
        if "messages" in chunk["output"] and chunk["output"]["messages"]:
            print(f"  Bot replies (Thread 2): {chunk['output']['messages'][-1].content}")
            assert "bob" in chunk['output']['messages'][-1].content.lower()
            assert "chef" in chunk['output']['messages'][-1].content.lower()
            print("  Assertion Passed: Bot recalled Bob's name and job.")

    # --- Verify Alice's memory is NOT contaminated by Bob ---
    print("\n--- Verification: Alice's memory integrity ---")
    thread_alice_3 = client.threads.create()
    print(f"  New Thread ID for Alice (3): {thread_alice_3.id}")

    print("  Alice says: 'Do you know my job?' (should not be mixed with Bob's)")
    response_stream = client.threads.invoke(
        thread_id=thread_alice_3.id,
        assistant_id=MAIN_CHATBOT_ASSISTANT_ID,
        input={"messages": ("human", "Do you know my job?")},
        config={"configurable": {"user_id": user_alice_id}}
    )
    async for chunk in response_stream:
        if "messages" in chunk["output"] and chunk["output"]["messages"]:
            print(f"  Bot replies (Thread 3): {chunk['output']['messages'][-1].content}")
            assert "data scientist" in chunk['output']['messages'][-1].content.lower()
            assert "chef" not in chunk['output']['messages'][-1].content.lower() # Crucial check
            print("  Assertion Passed: Alice's job recalled correctly, not confused with Bob's.")


    print("\n--- All tests completed. Check LangGraph Studio UI for memory documents under 'test_user_alice_123' and 'test_user_bob_456' ---")


if __name__ == "__main__":
    asyncio.run(test_user_memory_recall())