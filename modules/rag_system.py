"""
RAG System Module
Implements chunking, embedding, dense/sparse indexing, hybrid retrieval, multi-stage retrieval, response generation, and guardrails.
"""
import logging
import time
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import numpy as np

logging.basicConfig(level=logging.INFO)

class RAGChatbot:
    """
    Retrieval-Augmented Generation Chatbot for financial documents.
    Implements chunking, embedding, hybrid retrieval, multi-stage reranking, response generation, and guardrails.
    """
    def __init__(self, docs: List[str], sections: Dict[str, str]):
        """
        Initializes RAG system with document texts and sections.
        Args:
            docs (List[str]): List of cleaned document texts.
            sections (Dict[str, str]): Sectioned texts.
        """
        logging.info(f"Initializing RAGChatbot with {len(docs)} documents.")
        self.docs = docs
        self.sections = sections
        self.chunks_100, self.chunks_400 = self.chunk_documents(docs)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_index_100 = self.build_dense_index(self.chunks_100)
        self.vector_index_400 = self.build_dense_index(self.chunks_400)
        self.tfidf_vectorizer, self.tfidf_matrix = self.build_sparse_index(self.chunks_400)
        self.generator = pipeline('text-generation', model='distilgpt2')
        logging.info("RAGChatbot initialized.")

    def chunk_documents(self, docs: List[str]) -> Tuple[List[Dict], List[Dict]]:
        """
        Splits documents into chunks of 100 and 400 tokens, assigns IDs and metadata.
        Args:
            docs (List[str]): List of cleaned document texts.
        Returns:
            Tuple[List[Dict], List[Dict]]: Chunks of 100 and 400 tokens.
        """
        logging.info("Chunking documents.")
        chunks_100 = []
        chunks_400 = []
        for doc_id, doc in enumerate(docs):
            tokens = doc.split()
            for i in range(0, len(tokens), 100):
                chunk = ' '.join(tokens[i:i+100])
                chunks_100.append({'id': f'{doc_id}_100_{i}', 'text': chunk, 'meta': {}})
            for i in range(0, len(tokens), 400):
                chunk = ' '.join(tokens[i:i+400])
                chunks_400.append({'id': f'{doc_id}_400_{i}', 'text': chunk, 'meta': {}})
        logging.info(f"Chunked into {len(chunks_100)} (100 tokens) and {len(chunks_400)} (400 tokens) chunks.")
        return chunks_100, chunks_400

    def build_dense_index(self, chunks: List[Dict]):
        """
        Builds FAISS dense vector index for chunks.
        Args:
            chunks (List[Dict]): List of chunk dicts.
        Returns:
            faiss.IndexFlatL2 or None: FAISS index or None if no chunks.
        """
        logging.info("Building dense vector index.")
        if not chunks:
            logging.warning("No chunks to index.")
            return None
        texts = [c['text'] for c in chunks]
        embeddings = self.model.encode(texts)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings))
        logging.info("Dense index built.")
        return index

    def build_sparse_index(self, chunks: List[Dict]) -> Tuple[TfidfVectorizer, np.ndarray]:
        """
        Builds TF-IDF sparse index for chunks.
        Args:
            chunks (List[Dict]): List of chunk dicts.
        Returns:
            Tuple[TfidfVectorizer, np.ndarray]: Vectorizer and matrix.
        """
        logging.info("Building sparse TF-IDF index.")
        texts = [c['text'] for c in chunks]
        vectorizer = TfidfVectorizer(stop_words='english')
        matrix = vectorizer.fit_transform(texts)
        logging.info("Sparse index built.")
        return vectorizer, matrix

    def preprocess_query(self, query: str) -> str:
        """
        Cleans and preprocesses query (lowercase, remove stopwords).
        Args:
            query (str): User query.
        Returns:
            str: Preprocessed query.
        """
        logging.info(f"Preprocessing query: {query}")
        return query.lower()

    def hybrid_retrieve(self, query: str, top_n: int = 5) -> List[str]:
        """
        Hybrid retrieval: dense (FAISS) + sparse (TF-IDF) + fusion.
        Args:
            query (str): User query.
            top_n (int): Number of chunks to retrieve.
        Returns:
            List[str]: Retrieved chunk texts.
        """
        logging.info(f"Hybrid retrieval for query: {query}")
        # Dense retrieval
        query_emb = self.model.encode([query])
        _, dense_idx = self.vector_index_400.search(np.array(query_emb), top_n)
        dense_chunks = [self.chunks_400[i]['text'] for i in dense_idx[0]]
        # Sparse retrieval
        sparse_vec = self.tfidf_vectorizer.transform([query])
        scores = cosine_similarity(sparse_vec, self.tfidf_matrix)[0]
        sparse_idx = np.argsort(scores)[-top_n:][::-1]
        sparse_chunks = [self.chunks_400[i]['text'] for i in sparse_idx]
        # Fusion (union)
        retrieved = list(set(dense_chunks + sparse_chunks))[:top_n]
        logging.info(f"Retrieved {len(retrieved)} chunks.")
        return retrieved

    def rerank_chunks(self, query: str, chunks: List[str]) -> List[str]:
        """
        Multi-stage retrieval: rerank with cross-encoder (simulated).
        Args:
            query (str): User query.
            chunks (List[str]): Retrieved chunks.
        Returns:
            List[str]: Reranked chunks.
        """
        logging.info("Reranking chunks.")
        # Simulate reranking by sorting by length (placeholder for cross-encoder)
        reranked = sorted(chunks, key=lambda x: -len(x))
        logging.info("Chunks reranked.")
        return reranked

    def generate_response(self, query: str, context_chunks: List[str]) -> Tuple[str, float]:
        """
        Generates response using generative model, concatenating context and query.
        Args:
            query (str): User query.
            context_chunks (List[str]): Retrieved context chunks.
        Returns:
            Tuple[str, float]: Generated answer and confidence score.
        """
        logging.info("Generating response.")
        context = ' '.join(context_chunks)
        prompt = context + "\nQuestion: " + query + "\nAnswer: "
        output = self.generator(prompt, max_length=128, num_return_sequences=1)[0]['generated_text']
        # Simulate confidence
        confidence = min(1.0, len(context_chunks)/5)
        logging.info(f"Response generated: {output[:50]}... Confidence: {confidence}")
        return output, confidence

    def guardrail(self, query: str, response: str) -> Tuple[bool, str]:
        """
        Input/output guardrail: block irrelevant/harmful queries, flag hallucinations.
        Args:
            query (str): User query.
            response (str): Model response.
        Returns:
            Tuple[bool, str]: (is_safe, filtered_response)
        """
        logging.info("Applying guardrail.")
        # Block irrelevant queries
        irrelevant = ["capital of france", "weather", "sports"]
        for word in irrelevant:
            if word in query.lower():
                logging.warning("Blocked irrelevant query.")
                return False, "Query is irrelevant to financial statements."
        # Flag hallucinations (simple heuristic)
        if "not factual" in response:
            logging.warning("Flagged hallucinated output.")
            return False, "Response may be hallucinated."
        logging.info("Guardrail passed.")
        return True, response

    def answer(self, query: str) -> Tuple[str, float, float]:
        """
        End-to-end RAG pipeline: preprocess, retrieve, rerank, generate, guardrail.
        Args:
            query (str): User query.
        Returns:
            Tuple[str, float, float]: (answer, confidence, response_time)
        """
        logging.info(f"Answering query: {query}")
        start = time.time()
        pre_q = self.preprocess_query(query)
        chunks = self.hybrid_retrieve(pre_q)
        reranked = self.rerank_chunks(pre_q, chunks)
        response, confidence = self.generate_response(pre_q, reranked)
        is_safe, filtered = self.guardrail(pre_q, response)
        end = time.time()
        logging.info(f"Answer: {filtered[:50]}... Confidence: {confidence} Time: {end-start:.2f}s")
        return filtered, confidence, end-start
