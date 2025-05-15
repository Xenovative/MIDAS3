import os
import time
import chromadb
from langchain.vectorstores import Chroma
from langchain.schema import Document
import numpy as np

# Ensure consistent paths with rag.py
CHROMA_PERSIST_DIR = os.path.join(os.getcwd(), "chroma_db")

def process_with_retry(func, args=None, kwargs=None, retry_wait=1, max_retries=3):
    """Process a function with retry logic.
    
    Args:
        func: The function to call
        args: List of args to pass
        kwargs: Dict of kwargs to pass
        retry_wait: How long to wait between retries
        max_retries: Maximum number of retries
        
    Returns:
        The result of the function call
    """
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
        
    retries = 0
    last_exception = None
    
    while retries <= max_retries:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            retries += 1
            if retries <= max_retries:
                print(f"Retry {retries}/{max_retries} after error: {e}")
                time.sleep(retry_wait)
                retry_wait *= 2  # Exponential backoff
            else:
                # Re-raise the last exception
                print(f"Giving up after {max_retries} retries. Last error: {e}")
                raise

def hybrid_search(query, collection_name, embedding_functions, k=100, fetch_k=1000, threshold=0.01, is_chinese=False):
    """Perform a hybrid search using multiple embedding models and retrieval strategies.
    
    This approach combines results from multiple embedding models and search strategies
    to maximize the chances of finding relevant documents, especially for Chinese queries.
    
    Args:
        query: The query to search for
        collection_name: The collection to search in
        embedding_functions: List of embedding functions to use
        k: Number of documents to return
        fetch_k: Number of candidates to fetch
        threshold: Similarity threshold
        is_chinese: Whether the query is in Chinese
        
    Returns:
        List of documents
    """
    print(f"Starting hybrid search for '{query}' in {collection_name}")
    
    # Create client and check collection
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    if collection_name not in [c.name for c in client.list_collections()]:
        print(f"Collection {collection_name} not found")
        return []
    
    # Store all results with deduplication
    all_results = {}
    primary_results = []
    
    # 1. Primary search with main embedding model
    try:
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_functions[0],
            persist_directory=CHROMA_PERSIST_DIR
        )
        
        # Set up retriever based on query language
        if is_chinese:
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": k,
                    "fetch_k": fetch_k,
                    "score_threshold": threshold,
                }
            )
        else:
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": k,
                    "fetch_k": fetch_k,
                    "lambda_mult": 0.7
                }
            )
        
        # Get results
        primary_results = process_with_retry(retriever.get_relevant_documents, [query], retry_wait=1, max_retries=3)
        print(f"Primary search found {len(primary_results)} results")
        
        # Add to results dict with hash to deduplicate
        for doc in primary_results:
            # Use content hash as key to deduplicate
            content_hash = hash(doc.page_content)
            if content_hash not in all_results:
                doc.metadata["retrieved_by"] = "primary"
                doc.metadata["retrieval_rank"] = 1.0  # Primary results get top rank
                all_results[content_hash] = doc
    except Exception as e:
        print(f"Primary search failed: {e}")
    
    # 2. Secondary searches with additional embedding models
    for i, embeddings in enumerate(embedding_functions[1:]):
        try:
            # Create vectorstore with secondary embeddings
            sec_vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=CHROMA_PERSIST_DIR
            )
            
            # Set up retriever
            sec_retriever = sec_vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": min(k, 50),  # Limit secondary results
                    "fetch_k": min(fetch_k, 500),
                    "score_threshold": threshold * 2  # Double threshold for secondary
                }
            )
            
            # Get results from secondary model
            sec_results = process_with_retry(sec_retriever.get_relevant_documents, [query], retry_wait=1, max_retries=2)
            print(f"Secondary search {i} found {len(sec_results)} results")
            
            # Add to results dict
            for doc in sec_results:
                content_hash = hash(doc.page_content)
                if content_hash not in all_results:
                    doc.metadata["retrieved_by"] = f"secondary_{i}"
                    doc.metadata["retrieval_rank"] = 0.8  # Secondary results get lower rank
                    all_results[content_hash] = doc
        except Exception as e:
            print(f"Secondary search {i} failed: {e}")
    
    # 3. For Chinese queries, also try keyword search as a last resort
    if is_chinese and len(primary_results) < 20:
        try:
            # Direct document fetch from ChromaDB
            chroma_collection = client.get_collection(collection_name)
            
            # Extract keywords - use individual characters and bigrams
            keywords = []
            # Add individual characters
            for char in query:
                if '\u4e00' <= char <= '\u9fff':
                    keywords.append(char)
            
            # Add bigrams
            for i in range(len(query)-1):
                bigram = query[i:i+2]
                if any('\u4e00' <= c <= '\u9fff' for c in bigram):
                    keywords.append(bigram)
            
            # For each keyword, try to find documents containing it
            for keyword in keywords[:5]:  # Limit to top 5 keywords
                try:
                    # Use the ChromaDB includes parameter
                    keyword_results = chroma_collection.query(
                        query_texts=[keyword],
                        n_results=10,
                        include=["documents", "metadatas"]
                    )
                    
                    # Process results
                    if keyword_results and 'documents' in keyword_results and keyword_results['documents']:
                        for idx, content in enumerate(keyword_results['documents'][0]):
                            # Create Document object
                            metadata = keyword_results['metadatas'][0][idx] if 'metadatas' in keyword_results else {}
                            metadata["retrieved_by"] = f"keyword_{keyword}"
                            metadata["retrieval_rank"] = 0.5  # Keyword results get lowest rank
                            
                            doc = Document(
                                page_content=content,
                                metadata=metadata
                            )
                            
                            content_hash = hash(content)
                            if content_hash not in all_results:
                                all_results[content_hash] = doc
                except Exception as kw_err:
                    print(f"Keyword search for '{keyword}' failed: {kw_err}")
        except Exception as e:
            print(f"Keyword search failed: {e}")
    
    # 4. If we still don't have enough results, try a full collection scan
    if len(all_results) < 10 and is_chinese:
        try:
            print("Attempting full collection scan as last resort")
            # Get up to 100 random documents
            peek_results = client.get_collection(collection_name).peek(limit=100)
            
            if peek_results and 'documents' in peek_results and peek_results['documents']:
                for idx, content in enumerate(peek_results['documents']):
                    metadata = peek_results['metadatas'][idx] if 'metadatas' in peek_results else {}
                    metadata["retrieved_by"] = "full_scan"
                    metadata["retrieval_rank"] = 0.3  # Full scan results get lowest rank
                    
                    doc = Document(
                        page_content=content,
                        metadata=metadata
                    )
                    
                    content_hash = hash(content)
                    if content_hash not in all_results:
                        all_results[content_hash] = doc
        except Exception as e:
            print(f"Full collection scan failed: {e}")
    
    # Convert results dict to list
    results_list = list(all_results.values())
    
    # Sort by retrieval rank (primary > secondary > keyword > full scan)
    results_list.sort(key=lambda x: x.metadata.get("retrieval_rank", 0), reverse=True)
    
    print(f"Hybrid search found {len(results_list)} total results after deduplication")
    return results_list
