# services/retriever.py
import numpy as np
from services.storage import VectorStore
from services.embedder import Embedder
import google.generativeai as genai
import os
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)


class Retriever:
    """
    Retrieves top relevant document chunks for a given query.
    """

    def __init__(self, index_path="data/faiss_index.bin", metadata_path="data/metadata.json",heading_boost=0.25):
        self.store = VectorStore(index_path=index_path, metadata_path=metadata_path)
        self.embedder = Embedder()
        self.model = SentenceTransformer("all-MiniLM-L6-v2") 
        self.heading_boost = heading_boost
        try:
            self.store.load()
        except FileNotFoundError as e:
            raise RuntimeError(f"❌ Vector store not found: {e}")

    def _heading_match_score(self, query: str, headings: List[str]) -> float:
        """
        Compute semantic similarity between query and each heading.
        Return max similarity (0 to 1).
        """
        if not headings:
            return 0.0
        query_emb = self.model.encode([query], convert_to_tensor=True)
        heading_embs = self.model.encode(headings, convert_to_tensor=True)
        sims = util.cos_sim(query_emb, heading_embs)[0].cpu().numpy()
        return float(np.max(sims))


    def expand_query(self,original_query: str) -> str:
        """
        Expands user query with synonyms and related legal/insurance terminology
        using Gemini LLM before retrieval.
        """
        prompt = f"""
        You are a domain expert in insurance and legal documents.
        Expand the following query by adding synonyms, related terms, and 
        possible alternative phrasings that may appear in the documents.
        Keep it short, comma-separated, and relevant.

        Example:
        Input: "Does this policy cover knee surgery?"
        Output: "knee surgery, orthopedic surgery, joint operation, surgical treatment for knee injury, knee replacement"

        Input query:
        {original_query}
        """

        try:
            model = genai.GenerativeModel("gemini-1.5-pro")
            response=model.generate_content(prompt)
            # print(response)
            expanded = response.candidates[0].content.parts[0].text.strip()
            return expanded
        except Exception as e:
            print(f"⚠️ Query expansion failed: {e}")
            return original_query

    def retrieve(self, query: str, top_k: int = 5):
        """
        Given a user query, retrieve top_k relevant chunks.
        """
        expanded_query = self.expand_query(query)
        query_embedding = self.embedder.get_embedding(expanded_query)
        if query_embedding is None:
            raise ValueError("❌ Failed to generate embedding for the query")

        results = self.store.search(query_embedding, top_k=top_k)

        boosted_results = []
        for r in results:
            base_score = -r["distance"]
            headings = r["metadata"].get("headings", [])
            heading_sim = self._heading_match_score(query, headings)
            boosted_score = base_score + (heading_sim * self.heading_boost)
            boosted_results.append({**r, "boosted_score": boosted_score})

        # Step 3: sort by boosted score and trim
        boosted_results.sort(key=lambda x: x["boosted_score"], reverse=True)
        return boosted_results[:top_k]

# if __name__ == "__main__":
#     retriever = Retriever()
#     query = "How is AI used in healthcare?"
#     top_chunks = retriever.retrieve(query, top_k=3)
    
#     print("\n🔍 Top Retrieved Chunks:")
#     for i, chunk in enumerate(top_chunks, 1):
#         print(f"\n{i}. Distance: {chunk['distance']:.4f}")
#         print(f"   Metadata: {chunk['metadata']}")
#         print(f"   Content: {chunk['chunk']}")
