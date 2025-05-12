import os
import uuid
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from config import OLLAMA_HOST

# Import custom loaders
from custom_loaders import ChineseTheologyXMLLoader, ChineseTheologyXMLDirectoryLoader

# Import progress tracker
from progress_tracker import create_progress, update_progress, get_progress

# --- Configuration ---
DEFAULT_EMBED_MODEL = "nomic-embed-text"
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "db", "chroma_db")
DEFAULT_COLLECTION_NAME = "rag_documents"
CHUNK_SIZE = 2000  # Increased from 1000 for more context per chunk
CHUNK_OVERLAP = 400  # Increased overlap to maintain context between chunks

# Function to get conversation-specific collection name
def get_conversation_collection_name(conversation_id):
    """Generate a collection name for a specific conversation"""
    return f"conversation_{conversation_id}"

# Ensure ChromaDB persistence directory exists
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# --- LangChain Embedding Function ---
# Use a model that better handles Chinese text
DEFAULT_EMBED_MODEL = "herald/dmeta-embedding-zh" if "herald/dmeta-embedding-zh" in os.environ.get('AVAILABLE_EMBEDDING_MODELS', DEFAULT_EMBED_MODEL) else DEFAULT_EMBED_MODEL

# Set up local file store for caching embeddings
store = LocalFileStore("./cache/embeddings")

# Create base embeddings model
base_embeddings = OllamaEmbeddings(
    base_url=OLLAMA_HOST,
    model=DEFAULT_EMBED_MODEL
)

# Create cached embeddings to improve performance
ollama_ef = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=base_embeddings,
    document_embedding_cache=store,
    namespace=base_embeddings.model
)

# --- Document Loading and Processing ---

def load_documents(source_directory):
    """Loads documents from various file types in the specified directory."""
    print(f"Loading documents from: {source_directory}")
    documents = []
    
    loader_txt = DirectoryLoader(source_directory, glob="**/*.txt", loader_cls=TextLoader, recursive=True, show_progress=True)
    documents.extend(loader_txt.load())
    
    loader_pdf = DirectoryLoader(source_directory, glob="**/*.pdf", loader_cls=PyPDFLoader, recursive=True, show_progress=True)
    documents.extend(loader_pdf.load())

    loader_md = DirectoryLoader(source_directory, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader, recursive=True, show_progress=True)
    documents.extend(loader_md.load())
    
    # Use custom XML loader for Chinese theology documents
    try:
        print("Loading XML files with custom Chinese theology loader")
        # Use our custom directory loader that understands the XML structure
        xml_loader = ChineseTheologyXMLDirectoryLoader(
            source_directory,
            glob_pattern="**/*.xml",
            recursive=True
        )
        xml_docs = xml_loader.load()
        print(f"Loaded {len(xml_docs)} structured documents from XML files")
        documents.extend(xml_docs)
    except Exception as e:
        print(f"Error loading XML files with custom loader: {e}")
        # Fallback to trying individual files
        try:
            print("Falling back to individual file processing")
            import glob
            xml_files = glob.glob(os.path.join(source_directory, "**/*.xml"), recursive=True)
            for xml_file in xml_files:
                try:
                    loader = ChineseTheologyXMLLoader(xml_file)
                    file_docs = loader.load()
                    print(f"Loaded {len(file_docs)} documents from {os.path.basename(xml_file)}")
                    documents.extend(file_docs)
                except Exception as file_error:
                    print(f"Error processing {xml_file}: {file_error}")
        except Exception as e2:
            print(f"Error with fallback XML processing: {e2}")
            # Continue without XML files if both methods fail

    print(f"Loaded {len(documents)} documents.")
    return documents

def split_documents(documents, operation_id=None):
    """Splits documents into smaller chunks with enhanced handling for Chinese text.
    Uses parallel processing for large document sets."""
    print("Splitting documents...")
    if operation_id:
        update_progress(operation_id, description="Analyzing documents for optimal splitting")
    
    # Check if any document contains Chinese text
    has_chinese = False
    for doc in documents:
        if any('\u4e00' <= char <= '\u9fff' for char in doc.page_content):
            has_chinese = True
            break
    
    if has_chinese:
        print("Chinese text detected - using specialized splitter")
        # For Chinese text, we need different separators
        # Chinese doesn't use spaces between words, so we need to split on punctuation
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["。", "！", "？", "；", "\n\n", "\n", "，", ".", "!", "?", ";", ",", " ", ""],
            keep_separator=True
        )
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
    
    # For large document sets, use parallel processing
    if len(documents) > 50:
        import concurrent.futures
        import multiprocessing
        import math
        
        # Determine optimal number of workers based on CPU cores
        num_cores = multiprocessing.cpu_count()
        num_workers = max(2, min(num_cores - 1, 8))  # Use up to N-1 cores, max 8 workers
        
        print(f"Using {num_workers} parallel workers for document splitting on {num_cores} CPU cores")
        if operation_id:
            update_progress(operation_id, description=f"Parallel splitting with {num_workers} workers",
                          details={"workers": num_workers, "documents": len(documents)})
        
        # Split documents into batches for parallel processing
        batch_size = math.ceil(len(documents) / num_workers)
        batches = [documents[i:i+batch_size] for i in range(0, len(documents), batch_size)]
        
        all_splits = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Process each batch in parallel
            future_to_batch = {executor.submit(text_splitter.split_documents, batch): i 
                              for i, batch in enumerate(batches)}
            
            # Collect results as they complete
            for i, future in enumerate(concurrent.futures.as_completed(future_to_batch)):
                batch_index = future_to_batch[future]
                try:
                    batch_splits = future.result()
                    all_splits.extend(batch_splits)
                    if operation_id:
                        progress = (i + 1) / len(batches) * 100
                        update_progress(operation_id, progress=progress,
                                      description=f"Processed batch {i+1}/{len(batches)}",
                                      details={"batch": i+1, "total_batches": len(batches),
                                              "chunks_so_far": len(all_splits)})
                except Exception as e:
                    print(f"Error processing batch {batch_index}: {e}")
        
        split_docs = all_splits
    else:
        # For smaller document sets, use single-threaded processing
        split_docs = text_splitter.split_documents(documents)
    
    print(f"Split into {len(split_docs)} chunks.")
    if operation_id:
        update_progress(operation_id, description=f"Completed splitting into {len(split_docs)} chunks")
    return split_docs

# --- Vector Store Management ---

def setup_vector_store(source_directory, collection_name=DEFAULT_COLLECTION_NAME):
    """Loads, splits, and indexes documents into the vector store using LangChain Chroma class."""
    print("--- Setting up Vector Store using LangChain Chroma ---")
    documents = load_documents(source_directory)
    if not documents:
        print("No documents found in source directory.")
        return None # Return None if no documents
        
    split_docs = split_documents(documents)
    
    print(f"Creating/loading Chroma vector store at: {CHROMA_PERSIST_DIR}")
    # Use Chroma class to handle persistence and embedding
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=ollama_ef, # Pass LangChain embedding function
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=collection_name
    )
    
    print(f"Persisting vector store to: {CHROMA_PERSIST_DIR}")
    vectorstore.persist() 
    print("--- Vector Store Setup Complete ---")
    return vectorstore # Return the LangChain vectorstore object

# --- Function to add a single document --- 
def add_single_document_to_store(file_path, collection_name=DEFAULT_COLLECTION_NAME, conversation_id=None):
    """Loads, splits, and adds a single document file to the vector store.
    
    Args:
        file_path: Path to the document file
        collection_name: Name of the collection to add the document to
        conversation_id: If provided, will use a conversation-specific collection
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # If conversation_id is provided, use a conversation-specific collection
        if conversation_id:
            collection_name = get_conversation_collection_name(conversation_id)
            
        print(f"--- Adding single document: {file_path} to collection: {collection_name} ---")
        filename = os.path.basename(file_path)
        _, file_extension = os.path.splitext(filename)
        file_extension = file_extension.lower()

        # Load the document based on file extension
        try:
            if file_extension == '.txt':
                loader = TextLoader(file_path)
            elif file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension == '.md':
                loader = UnstructuredMarkdownLoader(file_path)
            elif file_extension == '.xml':
                # Use our custom XML loader for better handling of Chinese theology documents
                loader = ChineseTheologyXMLLoader(file_path)
                print(f"Using custom XML loader for {filename}")
            else:
                print(f"Unsupported file type: {file_extension}. Skipping.")
                return False

            documents = loader.load()
            if not documents:
                print(f"Could not load document: {filename}")
                return False
                
            # Split the document
            split_docs = split_documents(documents) # Use existing split function
            if not split_docs:
                print(f"Could not split document: {filename}")
                return False
                
            # Add chunks to vector store
            vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=ollama_ef,
                persist_directory=CHROMA_PERSIST_DIR,
                collection_name=collection_name
            )
            vectorstore.persist()
            print(f"Added {len(split_docs)} chunks to vector store collection '{collection_name}'")
            return True
        except Exception as e:
            print(f"Error processing document {filename}: {e}")
            return False
            

    except Exception as e:
        print(f"Error adding single document {file_path}: {e}")
        return False

# --- RAG Query Function ---

def has_documents(collection_name=DEFAULT_COLLECTION_NAME, conversation_id=None):
    """Checks if any documents exist in the vector store.
    
    Args:
        collection_name: Name of the collection to check
        conversation_id: If provided, will check a conversation-specific collection
        
    Returns:
        bool: True if documents exist, False otherwise
    """
    try:
        # If conversation_id is provided, use a conversation-specific collection
        if conversation_id:
            collection_name = get_conversation_collection_name(conversation_id)
            
        # Check if the Chroma DB directory exists
        if not os.path.exists(CHROMA_PERSIST_DIR):
            print(f"Vector store directory does not exist: {CHROMA_PERSIST_DIR}")
            return False
            
        # Load the persisted vector store
        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR, 
            embedding_function=ollama_ef, 
            collection_name=collection_name
        )
        
        # Get the collection and check if it has any documents
        collection = vectorstore._collection
        count = collection.count()
        
        print(f"Vector store collection '{collection_name}' has {count} documents.")
        return count > 0
    except Exception as e:
        print(f"Error checking for documents in vector store: {e}")
        return False

def retrieve_context(query, collection_name=DEFAULT_COLLECTION_NAME, conversation_id=None, n_results=500, operation_id=None):
    """Retrieves relevant document chunks for a given query using LangChain Chroma.
    
    Args:
        query: The query to search for
        collection_name: Name of the collection to search in
        conversation_id: If provided, will search in a conversation-specific collection
        n_results: Number of results to return
        
    Returns:
        str: The retrieved context as a string
    """
    try:
        # Generate operation_id if not provided
        if operation_id is None:
            operation_id = f"rag_{uuid.uuid4().hex[:8]}"
            
        # Initialize progress tracking
        create_progress(
            operation_id=operation_id,
            total_steps=10,  # We'll divide the retrieval process into 10 steps
            description=f"Retrieving context for query: '{query[:30]}...'"
        )
        update_progress(operation_id, current_step=1, description="Initializing retrieval")
        
        # If conversation_id is provided, use a conversation-specific collection
        if conversation_id:
            collection_name = get_conversation_collection_name(conversation_id)
            
        print(f"--- Retrieving Context from Collection: {collection_name} ---")
        update_progress(operation_id, current_step=2, description=f"Connecting to collection: {collection_name}")
        
        # Initialize the Chroma vector store with the specified collection
        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR, 
            embedding_function=ollama_ef, 
            collection_name=collection_name
        )
        update_progress(operation_id, current_step=3, description="Connected to vector database")
        
        # Get total document count to determine retrieval strategy
        total_docs = vectorstore._collection.count()
        print(f"Total documents in collection: {total_docs}")
        update_progress(operation_id, current_step=4, 
                       description=f"Found {total_docs} documents in collection",
                       details={"total_documents": total_docs})
        
        # Determine how many documents to retrieve based on collection size
        # For all collections, try to retrieve a substantial portion of documents
        if total_docs < 5000:
            # For small collections, retrieve almost everything
            fetch_k = min(total_docs, 500)  # Retrieve up to 500 docs for small collections
            k_value = min(total_docs, n_results)
        elif total_docs < 20000:
            # For medium collections, retrieve a large portion
            fetch_k = min(int(total_docs * 0.3), 500)  # Up to 30% of docs, max 500
            k_value = min(int(total_docs * 0.3), n_results)  # Up to 30% of docs, max n_results
        else:
            # For very large collections, still retrieve a significant amount
            fetch_k = 500  # Fixed large value
            k_value = n_results
        
        print(f"Retrieval strategy: fetch_k={fetch_k}, k={k_value}")
        update_progress(operation_id, current_step=5, 
                       description=f"Planning to retrieve {k_value} documents (fetch_k={fetch_k})",
                       details={"fetch_k": fetch_k, "k_value": k_value})
        
        # For Chinese text, we need to be more lenient with similarity scores
        # Use MMR retriever to maximize relevance and diversity
        update_progress(operation_id, current_step=6, 
                       description="Setting up retriever with MMR search strategy",
                       details={"search_type": "mmr", "lambda_mult": 0.7})
        
        retriever = vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance - balances relevance with diversity
            search_kwargs={
                "k": k_value,
                "fetch_k": fetch_k,  # Fetch more candidates to filter from
                "lambda_mult": 0.7  # Lower values (0.5-0.7) prioritize diversity over pure relevance
            }
        )
        update_progress(operation_id, current_step=7, 
                       description="Retriever configured, preparing to execute query")
        
        # Preprocess query to improve retrieval effectiveness
        # 1. Detect language
        is_chinese = any('\u4e00' <= char <= '\u9fff' for char in query)
        
        # 2. Remove stopwords and normalize
        import re
        from string import punctuation
        
        # Simple preprocessing function
        def preprocess_query(text, is_chinese=False):
            # For non-Chinese text
            if not is_chinese:
                # Convert to lowercase
                text = text.lower()
                # Remove punctuation
                text = re.sub(f'[{re.escape(punctuation)}]', ' ', text)
                # Remove extra whitespace
                text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        # Apply preprocessing
        processed_query = preprocess_query(query, is_chinese)
        
        print(f"Querying vector store for: '{query[:50]}...'")
        print(f"Query language detected: {'Chinese' if is_chinese else 'Other'}")
        print(f"Processed query: '{processed_query[:50]}...'")
        
        update_progress(operation_id, current_step=8, 
                       description=f"Executing query: '{query[:30]}...'" + (" (Chinese text detected)" if is_chinese else ""),
                       details={"is_chinese": is_chinese, "query_sample": query[:50], "processed_query": processed_query[:50]})
        print(f"Collection: {collection_name}, Documents: {vectorstore._collection.count()}")
        
        # Extract key terms for Chinese queries
        key_terms = []
        if is_chinese:
            # Extract all Chinese characters as potential key terms
            key_terms = [char for char in query if '\u4e00' <= char <= '\u9fff']
            print(f"Extracted {len(key_terms)} Chinese characters: {''.join(key_terms)}")
            
            # Look for multi-character terms (basic approach)
            # In Chinese, meaningful terms are often 2-4 characters
            multi_char_terms = []
            for i in range(len(key_terms) - 1):
                # Add 2-character terms
                multi_char_terms.append(key_terms[i] + key_terms[i+1])
                
                # Add 3-character terms if possible
                if i < len(key_terms) - 2:
                    multi_char_terms.append(key_terms[i] + key_terms[i+1] + key_terms[i+2])
            
            print(f"Generated {len(multi_char_terms)} multi-character terms: {', '.join(multi_char_terms)}")
            
        # Use the retriever to get relevant documents with multiple strategies in parallel
        all_results = []
        try:
            update_progress(operation_id, current_step=9, 
                           description="Starting document retrieval with multiple strategies")
            
            import concurrent.futures
            
            # Define multiple retrieval strategies as functions
            def strategy1():
                print("Strategy 1: Using processed query")
                return retriever.get_relevant_documents(processed_query)
                
            def strategy2():
                if is_chinese:
                    print("Strategy 2: Using key terms for Chinese query")
                    # Join terms with OR for broader matching
                    term_query = " OR ".join(multi_char_terms[:8])  # Limit to top terms
                    return retriever.get_relevant_documents(term_query)
                return []
                
            def strategy3():
                if is_chinese and len(query) > 20:
                    print("Strategy 3: Using query summary")
                    # Use first 20 chars as a summary query
                    return retriever.get_relevant_documents(query[:20])
                return []
            
            import multiprocessing
            num_cores = multiprocessing.cpu_count()
            max_workers = min(num_cores, 4)  # Use up to 4 cores for retrieval
            
            print(f"Using {max_workers} parallel workers for retrieval strategies")
            update_progress(operation_id, current_step=9.5, 
                           description=f"Running {max_workers} parallel retrieval strategies",
                           details={"workers": max_workers, "cpu_cores": num_cores})
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all strategies
                futures = []
                futures.append(executor.submit(strategy1))
                if is_chinese:
                    futures.append(executor.submit(strategy2))
                    if len(query) > 20:
                        futures.append(executor.submit(strategy3))
                
                # Process results as they complete
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    try:
                        results = future.result()
                        all_results.extend(results)
                        print(f"Strategy {i+1} results: {len(results)}")
                        update_progress(operation_id, 
                                      description=f"Retrieved {len(results)} documents from strategy {i+1}",
                                      details={"strategy": i+1, "results": len(results)})
                    except Exception as e:
                        print(f"Error in retrieval strategy {i+1}: {e}")
            
            # Add a final strategy for direct similarity search if needed
            if len(all_results) < 5:  # If we still don't have enough results
                def strategy_direct():
                    print("Strategy Direct: Using direct similarity search")
                    return vectorstore.similarity_search(query, k=n_results)
                
                try:
                    results_direct = strategy_direct()
                    all_results.extend(results_direct)
                    print(f"Direct similarity search results: {len(results_direct)}")
                except Exception as e:
                    print(f"Error in direct similarity search: {e}")
            
            # Deduplicate and limit results using memory-optimized approach
            # Use a more sophisticated deduplication strategy that leverages EC2 memory
            update_progress(operation_id, current_step=9.8, 
                           description="Optimizing and deduplicating results",
                           details={"total_candidates": len(all_results)})
            
            # Use a dictionary for faster lookups
            unique_docs = {}
            for doc in all_results:
                # Create a more robust hash key using both content and metadata
                content_sample = doc.page_content[:150] if len(doc.page_content) > 150 else doc.page_content
                source = doc.metadata.get('source', '')
                # Combine content and source for a more precise deduplication key
                dedup_key = f"{source}:{content_sample}"
                
                # If we haven't seen this document before, or if it's a better match
                if dedup_key not in unique_docs:
                    unique_docs[dedup_key] = doc
            
            # Convert back to list and limit results
            results = list(unique_docs.values())
            
            # Sort by relevance if we have metadata with scores
            if results and hasattr(results[0], 'metadata') and 'score' in results[0].metadata:
                results.sort(key=lambda x: x.metadata.get('score', 0), reverse=True)
                
            # Limit to requested number
            if len(results) > n_results:
                results = results[:n_results]
                    
            print(f"Final results after deduplication: {len(results)}")
            
            # If we still have no results, fall back to basic similarity search
            if not results:
                print("No results from all strategies, falling back to basic similarity search")
                results = vectorstore.similarity_search(query, k=n_results)
        
        except Exception as e:
            print(f"Error during retrieval: {e}")
            # Fallback to basic similarity search
            results = vectorstore.similarity_search(query, k=n_results)
            print(f"Fallback results: {len(results)}")
        
        # If we still have no results, try one last desperate approach
        if not results and is_chinese:
            print("Last resort: Searching for any Chinese content")
            # Try to find any documents with Chinese text
            for char in key_terms:
                try:
                    char_results = vectorstore.similarity_search(char, k=5)
                    if char_results:
                        print(f"Found {len(char_results)} results for character '{char}'")
                        results.extend(char_results)
                        if len(results) >= n_results:
                            results = results[:n_results]
                            break
                except Exception:
                    continue
        
        print(f"Retrieved {len(results)} context chunks from knowledge base")
        
        # Format the context for the model with better organization and relevance scoring
        formatted_context = ""
        
        # Score the results based on relevance indicators
        scored_results = []
        for i, doc in enumerate(results):
            # Extract source document name from metadata if available
            source = doc.metadata.get('source', 'Unknown source')
            # Get just the filename without path
            if source != 'Unknown source' and isinstance(source, str):
                source = os.path.basename(source)
            
            # Calculate a basic relevance score
            # This is a simple heuristic - could be improved with proper relevance scoring
            score = 1.0
            
            # Check if query terms appear in the content (basic relevance check)
            if is_chinese:
                # For Chinese, check if any of the multi-character terms appear
                if 'multi_char_terms' in locals() and multi_char_terms:
                    for term in multi_char_terms[:5]:  # Check the first 5 terms
                        if term in doc.page_content:
                            score += 0.2  # Boost score for each matching term
            else:
                # For non-Chinese, check for query words
                query_words = query.lower().split()
                for word in query_words:
                    if len(word) > 3 and word.lower() in doc.page_content.lower():
                        score += 0.1
            
            # Penalize very short or very long chunks slightly
            content_len = len(doc.page_content)
            if content_len < 100:
                score -= 0.1
            elif content_len > 3000:
                score -= 0.1
                
            scored_results.append((doc, source, score))
        
        # Sort by score (highest first)
        scored_results.sort(key=lambda x: x[2], reverse=True)
        
        # Format the context, now ordered by relevance and with rich metadata
        for i, (doc, source, score) in enumerate(scored_results):
            # Extract rich metadata if available
            title = doc.metadata.get('title', 'Unknown Title')
            author = doc.metadata.get('author', 'Unknown Author')
            periodical = doc.metadata.get('periodical', '')
            doc_type = doc.metadata.get('type', 'content')
            
            # Build a header with the metadata
            header = f"--- Context Chunk {i+1} (Relevance: {score:.2f}) ---\n"
            if title != 'Unknown Title':
                header += f"Title: {title}\n"
            if author != 'Unknown Author':
                header += f"Author: {author}\n"
            if periodical:
                header += f"Publication: {periodical}\n"
            header += f"Source: {source}\n"
            
            # Format each chunk with rich metadata and content
            formatted_context += f"\n\n{header}\n{doc.page_content}\n"
        
        # Add a comprehensive summary of what was retrieved
        if results:
            # Calculate the percentage of the knowledge base that was retrieved
            if 'total_docs' in locals():
                retrieval_percentage = (len(results) / total_docs) * 100 if total_docs > 0 else 0
                context_summary = f"\n\nRetrieved {len(results)} relevant context chunks ({retrieval_percentage:.1f}% of knowledge base) "
            else:
                context_summary = f"\n\nRetrieved {len(results)} relevant context chunks from knowledge base "
                
            # Add query information
            if is_chinese:
                context_summary += f"for Chinese query: '{query}'"
            else:
                context_summary += f"for query: '{query}'"
                
            # Add document coverage information
            sources = set()
            for doc, source, _ in scored_results:
                if source != 'Unknown source':
                    sources.add(source)
            
            if sources:
                context_summary += f"\nSources consulted: {len(sources)} documents"
                
            formatted_context = context_summary + formatted_context
        
        # Print summary for logging
        if results:
            print(f"Retrieved {len(results)} document chunks, formatted context length: {len(formatted_context)}")
            update_progress(operation_id, current_step=10, status="completed",
                           description=f"Retrieved {len(results)} document chunks",
                           details={
                               "chunks_retrieved": len(results),
                               "context_length": len(formatted_context),
                               "sources": len(set(doc.metadata.get('source', 'Unknown') for doc in results))
                           })
        else:
            print("No relevant context found in knowledge base")
            update_progress(operation_id, current_step=10, status="completed",
                           description="No relevant context found in knowledge base")
            
        return formatted_context
    except Exception as e:
        error_msg = f"Error retrieving context from Chroma vector store: {e}"
        print(error_msg)
        print("Please ensure the vector store has been set up correctly.")
        
        # Update progress with error status
        if 'operation_id' in locals():
            update_progress(operation_id, status="error", error=str(e),
                          description="Error during context retrieval")
        
        return ""

# --- Example Usage (Optional) ---
if __name__ == "__main__":
    docs_dir = "docs" 
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        with open(os.path.join(docs_dir, "test.txt"), "w") as f:
            f.write("This is a test document for the RAG system.")
            
    print(f"Running example setup with directory: '{docs_dir}'")
    # Setup function now returns the vectorstore object (or None)
    vs = setup_vector_store(docs_dir)
    
    # Only test retrieval if setup was successful
    if vs:
        print("\n--- Testing Retrieval ---")
        test_query = "What is the RAG system?"
        retrieved = retrieve_context(test_query)
        print(f"\nQuery: {test_query}")
        print(f"Retrieved Context:\n{retrieved}")

        test_query_2 = "non-existent topic"
        retrieved_2 = retrieve_context(test_query_2)
        print(f"\nQuery: {test_query_2}")
        print(f"Retrieved Context:\n{retrieved_2}")
    else:
        print("\nSkipping retrieval test as vector store setup failed.")
