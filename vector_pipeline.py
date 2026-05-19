import os
import uuid
import pandas as pd
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

class VectorStoragePipeline:
    def __init__(self, index_name: str = "nexusops-seo-index"):
        # Initialize API Clients
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        self.index_name = index_name
        self._initialize_pinecone_index()

    def _initialize_pinecone_index(self):
        """
        Idempotent index provisioning using Pinecone Serverless Spec.
        """
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=1536, # Matches text-embedding-3-small dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        self.index = self.pc.Index(self.index_name)

    def chunk_text(self, text: str, max_tokens: int = 500, overlap: int = 50) -> list:
        """
        Splits text into chunks to maintain semantic cohesion.
        """
        words = text.split()
        chunks = []
        for i in range(0, len(words), max_tokens - overlap):
            chunk = " ".join(words[i:i + max_tokens])
            chunks.append(chunk)
        return chunks

    def generate_embeddings(self, text_chunks: list) -> list:
        """
        Calls OpenAI API to convert string blocks into 1536-dimensional float arrays.
        """
        response = self.openai_client.embeddings.create(
            input=text_chunks,
            model="text-embedding-3-small"
        )
        return [data.embedding for data in response.data]

    def upsert_dataframe_to_vector_db(self, df: pd.DataFrame):
        """
        Iterates over the scraped DataFrame, chunks content, generates vectors,
        and pushes payloads to Pinecone with structural metadata.
        """
        upsert_payload = []
        
        for _, row in df.iterrows():
            if row['title'] == "FAILED":
                continue
                
            chunks = self.chunk_text(row['extracted_content'])
            embeddings = self.generate_embeddings(chunks)
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_id = f"{uuid.uuid4()}"
                
                metadata = {
                    "url": row['url'],
                    "title": row['title'],
                    "timestamp": row['timestamp'],
                    "text_content": chunk
                }
                
                upsert_payload.append((vector_id, embedding, metadata))
        
        if upsert_payload:
            self.index.upsert(vectors=upsert_payload)
            print(f"Successfully vectorized and provisioned {len(upsert_payload)} nodes into Pinecone.")

    def query_semantic_matches(self, user_query: str, top_k: int = 3) -> list:
        """
        Executes a real-time vector search using Cosine Similarity matching.
        """
        query_vector = self.generate_embeddings([user_query])[0]
        
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )
        return results['matches']

if __name__ == "__main__":
    # Mock data representing input from the Playwright scraper pipeline
    mock_scraped_data = pd.DataFrame([{
        "url": "https://example-competitor.com/seo-guide",
        "title": "Advanced Enterprise SEO Strategy",
        "extracted_content": "Artificial intelligence workflows are fundamentally transforming search engine optimization. Multi-agent architectures leverage vector databases like Pinecone to dynamically analyze user search patterns and automatically build content blueprints.",
        "timestamp": "2026-05-19T21:00:00Z"
    }])
    
    vector_pipeline = VectorStoragePipeline()
    vector_pipeline.upsert_dataframe_to_vector_db(mock_scraped_data)
    
    print("\n--- Testing Real-Time Semantic Search ---")
    matches = vector_pipeline.query_semantic_matches("How are AI systems changing search planners?")
    for match in matches:
        print(f"Score: {match['score']:.4f} | Source: {match['metadata']['title']}")
        print(f"Content: {match['metadata']['text_content']}\n")

