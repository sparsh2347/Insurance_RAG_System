# main.py
import os
import json
import hashlib
from dotenv import load_dotenv
from services.extraction import extract_text_from_file
from services.chunker import HybridChunker
from services.embedder import Embedder
from services.storage import VectorStore
from services.retriever import Retriever
from services.evaluator import Evaluator
PROCESSOR_ID=os.getenv("PROCESSOR_ID")
PROJECT_ID=os.getenv("PROJECT_ID")
load_dotenv()

INDEX_PATH = "data/faiss_index.bin"
METADATA_PATH = "data/metadata.json"
CACHE_FILE = "data/processed_docs.json"

def _load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def _save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

def _hash_doc_id(doc_path: str) -> str:
    return hashlib.sha256(doc_path.encode()).hexdigest()


def process_document(file_path: str):
    """
    Step 1-4: Extract text → Chunk → Embed → Store in vector DB
    """
    print(f"\n📂 Processing document: {file_path}")

     # Load cache
    cache = _load_cache()
    doc_hash = _hash_doc_id(file_path)

    # Skip if already processed
    if doc_hash in cache:
        print("⚡ Document already processed. Skipping reprocessing.")
        return
    
    # Extract text from PDF using Cloud Document AI
    text = extract_text_from_file(file_path,PROCESSOR_ID,PROJECT_ID)

    # Chunk the extracted text
    chunker = HybridChunker()
    raw_chunks = chunker.chunk(text)
    # chunks = [{"content": chunk.text, "metadata": chunk.metadata, "id":chunk.id, "index":chunk.index} for chunk in raw_chunks]

    # Embed chunks
    embedder = Embedder()
    embeddings = embedder.embed_chunks(raw_chunks)

    # Store embeddings + metadata
    store = VectorStore(index_path=INDEX_PATH, metadata_path=METADATA_PATH)
    store.add_embeddings(embeddings)
    store.save()

    # Save this document in cache
    cache[doc_hash] = file_path
    _save_cache(cache)

    print("✅ Document processed and stored successfully.")


def query_system(query: str, top_k: int = 5):
    """
    Step 5-6: Retrieve relevant chunks → Generate final answer
    """
    print(f"\n🔍 User Query: {query}")

    # Retrieve relevant chunks
    retriever = Retriever(index_path=INDEX_PATH, metadata_path=METADATA_PATH)
    retrieved_chunks = retriever.retrieve(query, top_k=top_k)

    # Evaluate & generate final answer
    evaluator = Evaluator()
    answer = evaluator.generate_answer(query, retrieved_chunks)

    print("\n🤖 Final Answer:\n")
    print(answer)
    return answer


# if __name__ == "__main__":
#     # === Example flow ===

#     # Step 1: Process documents (only once per document)
#     process_document("sample_files/sample.pdf")

#     # Step 2: Query the system
#     query_system("Does this policy cover knee surgery?", top_k=3)