"""
Test script for evaluating embedding quality and retrieval accuracy.
This script helps diagnose issues with embedding models and retrieval settings.
"""

import os
import sys
import time
import numpy as np
import logging

# Suppress unnecessary logs
logging.basicConfig(level=logging.ERROR)

# Suppress specific module logs
logging.getLogger('langchain').setLevel(logging.ERROR)
logging.getLogger('langchain_community').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.ERROR)

from langchain_community.embeddings import OllamaEmbeddings
from config import DEFAULT_EMBEDDING_MODEL, OLLAMA_HOST

# Test data with mixed language content
TEST_QUERIES = [
    # English only
    "What are the best books for learning Python programming?",
    "How to implement machine learning algorithms",
    "Software development best practices",
    
    # Chinese only
    "如何学习编程语言",
    "机器学习算法的实现方法",
    "软件开发最佳实践",
    
    # Mixed language
    "Python 编程 best practices",
    "机器学习 machine learning algorithms",
    "Software development 最佳实践"
]

# Test documents with mixed language content
TEST_DOCUMENTS = [
    # English only
    "Python is a high-level programming language known for its readability and versatility.",
    "Machine learning algorithms enable computers to learn from data and make predictions.",
    "Software development best practices include code reviews, testing, and documentation.",
    
    # Chinese only
    "Python是一种高级编程语言，以其可读性和多功能性而闻名。",
    "机器学习算法使计算机能够从数据中学习并做出预测。",
    "软件开发最佳实践包括代码审查、测试和文档编制。",
    
    # Mixed language
    "Python 是一种高级编程语言 known for its readability and versatility.",
    "Machine learning algorithms 使计算机能够从数据中学习并做出预测。",
    "Software development best practices 包括代码审查、测试和文档编制。"
]

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

def test_embedding_model(model_name=DEFAULT_EMBEDDING_MODEL):
    """Test embedding model quality with various language inputs"""
    print(f"\n=== Testing Embedding Model: {model_name} ===\n")
    
    # Suppress stdout temporarily during initialization
    import io
    from contextlib import redirect_stdout
    
    # Initialize embedding model
    print(f"Initializing embedding model from {OLLAMA_HOST}...")
    with redirect_stdout(io.StringIO()):
        embeddings = OllamaEmbeddings(
            base_url=OLLAMA_HOST,
            model=model_name,
            show_progress=False,  # Disable progress to reduce noise
            dimensions=1536  # Force consistent dimensions
        )
    
    # Test embedding dimensions
    print("\nTesting embedding dimensions...")
    test_text = "This is a test sentence for embedding dimension verification."
    test_embedding = embeddings.embed_query(test_text)
    print(f"Embedding dimensions: {len(test_embedding)}")
    
    # Test embedding quality for different languages
    print("\nTesting embedding quality for different languages...")
    
    # Embed all queries
    query_embeddings = []
    for i, query in enumerate(TEST_QUERIES):
        print(f"Embedding query {i+1}/{len(TEST_QUERIES)}: {query[:30]}...")
        query_embeddings.append(embeddings.embed_query(query))
    
    # Embed all documents
    doc_embeddings = []
    for i, doc in enumerate(TEST_DOCUMENTS):
        print(f"Embedding document {i+1}/{len(TEST_DOCUMENTS)}: {doc[:30]}...")
        doc_embeddings.append(embeddings.embed_query(doc))
    
    # Calculate similarity matrix
    print("\nCalculating similarity matrix...")
    similarity_matrix = np.zeros((len(TEST_QUERIES), len(TEST_DOCUMENTS)))
    
    for i, query_emb in enumerate(query_embeddings):
        for j, doc_emb in enumerate(doc_embeddings):
            similarity_matrix[i, j] = cosine_similarity(query_emb, doc_emb)
    
    # Print similarity matrix
    print("\n=== Similarity Matrix (Query x Document) ===")
    print("Higher values indicate better matches. Range: 0.0 to 1.0")
    print("\nQuery Types: [0-2] English, [3-5] Chinese, [6-8] Mixed")
    print("Document Types: [0-2] English, [3-5] Chinese, [6-8] Mixed\n")
    
    # Format header
    header = "Query\\Doc |"
    for j in range(len(TEST_DOCUMENTS)):
        header += f" Doc {j:2d} |"
    print(header)
    print("-" * len(header))
    
    # Print matrix with formatting
    for i in range(len(TEST_QUERIES)):
        row = f"Query {i:2d}  |"
        for j in range(len(TEST_DOCUMENTS)):
            # Highlight matches (same language or mixed)
            is_match = (i//3 == j//3) or (i//3 == 2) or (j//3 == 2)
            sim_val = similarity_matrix[i, j]
            
            if is_match and sim_val >= 0.7:
                # Good match
                row += f" \033[92m{sim_val:.2f}\033[0m |"  # Green
            elif is_match and sim_val >= 0.5:
                # Moderate match
                row += f" \033[93m{sim_val:.2f}\033[0m |"  # Yellow
            elif is_match:
                # Poor match
                row += f" \033[91m{sim_val:.2f}\033[0m |"  # Red
            else:
                # Non-match (expected low similarity)
                row += f" {sim_val:.2f} |"
        print(row)
    
    # Analyze results
    print("\n=== Analysis ===")
    
    # Check same-language retrieval
    eng_eng_avg = np.mean(similarity_matrix[0:3, 0:3])
    chi_chi_avg = np.mean(similarity_matrix[3:6, 3:6])
    mix_mix_avg = np.mean(similarity_matrix[6:9, 6:9])
    
    print(f"English-English average similarity: {eng_eng_avg:.4f}")
    print(f"Chinese-Chinese average similarity: {chi_chi_avg:.4f}")
    print(f"Mixed-Mixed average similarity: {mix_mix_avg:.4f}")
    
    # Check cross-language retrieval
    eng_chi_avg = np.mean(similarity_matrix[0:3, 3:6])
    chi_eng_avg = np.mean(similarity_matrix[3:6, 0:3])
    
    print(f"English-Chinese average similarity: {eng_chi_avg:.4f}")
    print(f"Chinese-English average similarity: {chi_eng_avg:.4f}")
    
    # Check mixed language retrieval
    mix_eng_avg = np.mean(similarity_matrix[6:9, 0:3])
    mix_chi_avg = np.mean(similarity_matrix[6:9, 3:6])
    
    print(f"Mixed-English average similarity: {mix_eng_avg:.4f}")
    print(f"Mixed-Chinese average similarity: {mix_chi_avg:.4f}")
    
    # Overall assessment
    print("\n=== Overall Assessment ===")
    
    if eng_eng_avg < 0.6:
        print("⚠️ WARNING: English-English similarity is low. Embedding model may not be optimal for English content.")
    
    if chi_chi_avg < 0.6:
        print("⚠️ WARNING: Chinese-Chinese similarity is low. Embedding model may not be optimal for Chinese content.")
    
    if mix_mix_avg < 0.6:
        print("⚠️ WARNING: Mixed-Mixed similarity is low. Embedding model may struggle with mixed language content.")
    
    language_bias = abs(eng_eng_avg - chi_chi_avg)
    if language_bias > 0.15:
        print(f"⚠️ WARNING: Significant language bias detected ({language_bias:.4f}). Model favors {'English' if eng_eng_avg > chi_chi_avg else 'Chinese'}.")
    
    # Recommendations
    print("\n=== Recommendations ===")
    
    if language_bias > 0.15:
        print("1. Consider using a more balanced embedding model for mixed language content.")
        print("   - Try models like 'mxbai-embed-large' or 'bge-large-zh' which handle both languages well.")
    
    if eng_eng_avg < 0.6 or chi_chi_avg < 0.6:
        print("2. Adjust retrieval parameters to compensate for lower similarity scores:")
        print("   - Lower the similarity threshold")
        print("   - Increase fetch_k and k values")
        print("   - Use hybrid search with BM25 to improve recall")
    
    print("3. For mixed language queries, consider:")
    print("   - Preprocessing queries to enhance both languages")
    print("   - Using multiple embedding models in parallel")
    print("   - Weighting English terms higher if they're being overshadowed")

if __name__ == "__main__":
    # Use default model or model specified as command line argument
    model_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EMBEDDING_MODEL
    test_embedding_model(model_name)
