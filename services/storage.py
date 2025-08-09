# services/vector_store.py
import os
import faiss
import numpy as np
import json

class VectorStore:
    def __init__(self, index_path="data/faiss_index.bin", metadata_path="data/metadata.json", dim=768):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)  # L2 distance search
        self.metadata = []  # store chunk info (source, token count, etc.)

    def add_embeddings(self, embedded_chunks):
        """
        Add embedded chunks to the FAISS index.
        """
        vectors = [chunk["embedding"] for chunk in embedded_chunks]
        vectors = np.array(vectors, dtype=np.float32)
        self.index.add(vectors)
        self.metadata.extend(embedded_chunks)

    def save(self):
        """
        Save FAISS index and metadata to disk.
        """
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved FAISS index to {self.index_path}")
        print(f"✅ Saved metadata to {self.metadata_path}")

    def load(self):
        """
        Load FAISS index and metadata from disk.
        """
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            print("✅ Vector store loaded successfully.")
        else:
            raise FileNotFoundError("❌ Index or metadata file not found.")

    def search(self, query_embedding, top_k=5):
        """
        Search the FAISS index for top_k similar chunks.
        """
        query_vector = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_vector, top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:
                results.append({
                    "chunk": self.metadata[idx]["text"],
                    "metadata": self.metadata[idx]["metadata"],
                    "distance": float(dist)
                })
        return results


# if __name__ == "__main__":
#     # Example usage
#     from services.embedder import Embedder

#     chunks = [
#         {"content": "AI is revolutionizing healthcare.", "metadata": {"source": "sample.pdf"}},
#         {"content": "Machine learning enables predictive analytics.", "metadata": {"source": "sample.pdf"}}
#     ]

#     embedder = Embedder()
#     embedded_chunks = embedder.embed_chunks(chunks)

#     store = VectorStore(dim=len(embedded_chunks[0]["embedding"]))
#     store.add_embeddings(embedded_chunks)
#     store.save()

#     # Load & search
#     store.load()
#     query = embedder.get_embedding("How is AI changing medicine?")
#     results = store.search(query, top_k=2)
#     for r in results:
#         print(r)
