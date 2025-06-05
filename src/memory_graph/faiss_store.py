import os
import faiss
import pickle
from typing import List
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.documents import Document

FAISS_DIR = "vector_store"

# Global embeddings model to be reused
embeddings_model = VertexAIEmbeddings(model="text-embedding-004")


def get_faiss_path(user_id: str, function_name: str) -> str:
    return os.path.join(FAISS_DIR, f"faiss_index_{user_id}_{function_name}")


def store_note_embedding(user_id: str, function_name: str, memory: dict) -> None:
    """Embed and store a single memory in FAISS."""
    content = memory.get("content", "")
    context = memory.get("context", "")
    if not content:
        print("WARNING: Attempted to store empty note content to FAISS.")
        return

    doc = Document(
        page_content=content,
        metadata={"context": context}
    )

    path = get_faiss_path(user_id, function_name)

    # Ensure the FAISS_DIR exists
    os.makedirs(FAISS_DIR, exist_ok=True)

    if os.path.exists(path):
        try:
            faiss_store = FAISS.load_local(
                path, embeddings_model, allow_dangerous_deserialization=True
            )
            faiss_store.add_documents([doc])
            print(f"DEBUG: Added document to existing FAISS index at: {path}")
        except Exception as e:
            print(f"ERROR: Could not load or add to existing FAISS index at {path}. Error: {e}. Creating new one.")
            faiss_store = FAISS.from_documents([doc], embeddings_model)
    else:
        faiss_store = FAISS.from_documents([doc], embeddings_model)
        print(f"DEBUG: Created new FAISS index at: {path}")

    faiss_store.save_local(path)
    print(f"DEBUG: FAISS index saved/updated at: {path}")


def search_faiss(user_id: str, function_name: str, query: str, k: int = 5) -> List[Document]:
    """Search the FAISS index for similar documents."""
    path = get_faiss_path(user_id, function_name)

    if not os.path.exists(path):
        print(f"DEBUG: FAISS index not found at {path}. Returning empty list.")
        return []

    try:
        faiss_store = FAISS.load_local(
            path, embeddings_model, allow_dangerous_deserialization=True
        )
        print(f"DEBUG: Searching FAISS index at: {path} with query: {query[:50]}")
        return faiss_store.similarity_search(query, k=k)
    except Exception as e:
        print(f"ERROR: Failed to load or search FAISS index at {path}. Error: {e}")
        return []
