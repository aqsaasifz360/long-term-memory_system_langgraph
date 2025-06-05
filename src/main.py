# src/main.py (or app.py)

import asyncio
from pathlib import Path

from langgraph.config import set_config

from langchain_google_vertexai import VertexAIEmbeddings

# Import your custom FAISS memory store
from memory_store.faiss_store import FAISSMemoryStore # Adjust import based on where src/ is relative to this file

# Import your chatbot graph and configuration
from chatbot.graph import graph as chatbot_graph
from chatbot.configuration import ChatConfigurable
from langgraph.client import LangGraphClient


# --- Configuration for FAISS Memory Store for LangGraph's Memory ---
# Define the path where the FAISS index for structured memories will be stored
# This should be distinct from your RAG FAISS index (if any, typically in vector_store/faiss_index)
FAISS_MEMORY_STORE_PATH = Path("./langgraph_memory_faiss_index") # This will be created in your project root
EMBEDDING_MODEL_FOR_MEMORIES = "text-embedding-004" # Or "gemini-2.0-flash" if you prefer

# Initialize the embedding model for the memory store
embeddings_for_memory = VertexAIEmbeddings(model_name=EMBEDDING_MODEL_FOR_MEMORIES)

# Create an instance of your custom FAISSMemoryStore
my_persistent_store = FAISSMemoryStore(
    embeddings=embeddings_for_memory,
    store_path=FAISS_MEMORY_STORE_PATH
)

# --- Set LangGraph's global store ---
# This is the crucial step to make memory persistent and searchable via FAISS.
# It MUST be called before you compile or run any LangGraph graphs that rely on `get_store()`.
set_config(store=my_persistent_store)

print(f"LangGraph configured to use FAISSMemoryStore at {FAISS_MEMORY_STORE_PATH}")

# --- Initialize and Run Your Chatbot ---
client = LangGraphClient(app=chatbot_graph)

async def run_chatbot_conversation(user_id: str):
    """Simulates a conversation with the chatbot, demonstrating memory persistence."""
    print(f"\n--- Starting conversation for user: {user_id} ---")

    # First interaction: User provides information
    config = {"configurable": {"user_id": user_id}}
    inputs = {"messages": ("human", "Hello, my name is Alice. I like hiking and drinking coffee. My job is a software engineer.")}

    print("\n[User]: Hello, my name is Alice. I like hiking and drinking coffee. My job is a software engineer.")
    async for s in client.astream(inputs, config=config):
        # The 'bot' node's output
        if "__end__" in s:
            print(f"[Bot]: {s['__end__']['messages'][-1].content}")
    
    print("\n--- Waiting for memory update (debounce) ---")
    # Allow time for scheduled memory update (debounce).
    # The delay_seconds is configured in chatbot/configuration.py
    await asyncio.sleep(ChatConfigurable.from_context().delay_seconds + 1)
    print("--- Memory update likely processed ---")

    # Second interaction: Ask about interests in a new "thread" (simulated by new input)
    # The user_id is the key for memory persistence.
    inputs = {"messages": ("human", "Can you remind me what my interests are?")}
    print("\n[User]: Can you remind me what my interests are?")
    async for s in client.astream(inputs, config=config):
        if "__end__" in s:
            print(f"[Bot]: {s['__end__']['messages'][-1].content}")
            
    # Third interaction: Update information
    inputs = {"messages": ("human", "Actually, I prefer tea over coffee now.")}
    print("\n[User]: Actually, I prefer tea over coffee now.")
    async for s in client.astream(inputs, config=config):
        if "__end__" in s:
            print(f"[Bot]: {s['__end__']['messages'][-1].content}")

    print("\n--- Waiting for memory update (debounce) ---")
    await asyncio.sleep(ChatConfigurable.from_context().delay_seconds + 1)
    print("--- Memory update likely processed ---")

    # Fourth interaction: Verify updated information
    inputs = {"messages": ("human", "So, what's my current beverage preference?")}
    print("\n[User]: So, what's my current beverage preference?")
    async for s in client.astream(inputs, config=config):
        if "__end__" in s:
            print(f"[Bot]: {s['__end__']['messages'][-1].content}")

    print(f"\n--- Conversation for user: {user_id} ended ---")


if __name__ == "__main__":
    # You can change the user ID here to test different users' memories
    test_user_id = "test-user-001"
    
    # Ensure environment variables for Google Vertex AI are set
    # e.g., GOOGLE_APPLICATION_CREDENTIALS, GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION
    # If using dotenv, load them here:
    # from dotenv import load_dotenv
    # load_dotenv() # Load your .env file

    asyncio.run(run_chatbot_conversation(test_user_id))
    # To test persistence, run the script, then stop it, and run it again
    # with the same test_user_id. The memories should be recalled.