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

def retrieve_context(query, collection_name=DEFAULT_COLLECTION_NAME, conversation_id=None, bot_collection=None, n_results=500, operation_id=None):
    """Retrieves relevant document chunks for a given query using LangChain Chroma.
    
    Args:
        query: The query to search for
        collection_name: Name of the collection to search in (default collection)
        conversation_id: If provided, will search in a conversation-specific collection
        bot_collection: If provided, will search in the bot's knowledge base collection
        n_results: Number of results to return
        operation_id: Unique identifier for tracking the retrieval process
        
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
        
        # Determine which collections to search based on our rules
        collections_to_search = []
        collection_descriptions = []
        
        # Rule 1: Conversation with a bot with knowledge base
        if bot_collection:
            collections_to_search.append(bot_collection)
            collection_descriptions.append(f"bot knowledge base ({bot_collection})")
            print(f"[Rule 1] Using bot knowledge base collection: {bot_collection}")
        
        # Rule 2 & 3: Conversation with documents uploaded
        if conversation_id:
            conv_collection = get_conversation_collection_name(conversation_id)
            # Check if this collection actually exists and has documents
            if has_documents(conversation_id=conversation_id):
                collections_to_search.append(conv_collection)
                collection_descriptions.append(f"conversation documents ({conv_collection})")
                print(f"[Rule 2/3] Using conversation documents collection: {conv_collection}")
        
        # If no specific collections were found, don't use ANY collection
        # This enforces our rule that RAG should only use specific collections based on our rules
        if not collections_to_search:
            print(f"No valid collections found based on our rules, not falling back to default")
            # Return empty results immediately to avoid any RAG search
            return ""
            
        # Log which collections we're searching
        collection_str = ', '.join(collections_to_search)
        print(f"--- Retrieving Context from Collections: {collection_str} ---")
        update_progress(operation_id, current_step=2, 
                       description=f"Searching {len(collections_to_search)} collections: {', '.join(collection_descriptions)}")
        
        # Initialize results list that will store results from all collections
        all_collection_results = []
        total_docs = 0
        update_progress(operation_id, current_step=3, description="Setting up vector database connections")
        
        # Function to process a single collection
        def process_collection(collection_name, search_percentage=0.2):
            try:
                print(f"Processing collection: {collection_name}")
                # Connect to the collection
                vectorstore = Chroma(
                    persist_directory=CHROMA_PERSIST_DIR, 
                    embedding_function=ollama_ef, 
                    collection_name=collection_name
                )
                
                # Get document count
                docs_in_collection = vectorstore._collection.count()
                print(f"Collection {collection_name} has {docs_in_collection} documents")
                
                if docs_in_collection == 0:
                    print(f"No documents in collection {collection_name}, skipping")
                    return []
                
                # Calculate how many documents to fetch based on collection size
                k_value = min(int(docs_in_collection * search_percentage), n_results)
                fetch_k = min(docs_in_collection, max(k_value * 2, 500))
                
                print(f"Retrieval strategy for {collection_name}: fetch_k={fetch_k}, k={k_value}")
                
                # Set up the retriever with MMR search
                retriever = vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": k_value,
                        "fetch_k": fetch_k,
                        "lambda_mult": 0.7
                    }
                )
                
                # Get documents from this collection
                results = retriever.get_relevant_documents(processed_query)
                print(f"Retrieved {len(results)} documents from collection {collection_name}")
                
                # Add metadata about which collection the results came from
                for doc in results:
                    if 'source_collection' not in doc.metadata:
                        doc.metadata['source_collection'] = collection_name
                
                return results
            except Exception as e:
                print(f"Error processing collection {collection_name}: {e}")
                return []
        
        # Search each collection based on our rules
        for i, collection_name in enumerate(collections_to_search):
            update_progress(operation_id, current_step=4, 
                           description=f"Searching collection {i+1}/{len(collections_to_search)}: {collection_name}")
            
            # Process this collection
            results = process_collection(collection_name)
            
            # Keep track of total documents across all collections
            if results:
                all_collection_results.extend(results)
                total_docs += len(results)
                print(f"Added {len(results)} results from {collection_name}, total now: {total_docs}")
        
        # Log results summary
        update_progress(operation_id, current_step=5, 
                       description=f"Found {total_docs} documents across {len(collections_to_search)} collections",
                       details={"total_documents": total_docs})
        
        # We've already retrieved documents from each collection separately
        # Now we need to process the combined results
        update_progress(operation_id, current_step=6, 
                       description="Processing and combining results from all collections")
            
        # Use all_collection_results as our combined results
        results = all_collection_results
            
        # No fallback approaches - we only use the collections specified by our rules
        if not results and collections_to_search:
            update_progress(operation_id, current_step=6.5, 
                           description="No results from search, but we don't use fallbacks anymore")
            print(f"No results found in specified collections, and fallbacks are disabled")
            # Empty results with no fallback
                
        # Update progress
        update_progress(operation_id, current_step=7, 
                       description=f"Retrieved a total of {len(results)} documents")
        
        # Preprocess query to improve retrieval effectiveness
        # 1. Detect language
        is_chinese = any('\u4e00' <= char <= '\u9fff' for char in query)
        
        # 2. Remove stopwords and normalize
        import re
        from string import punctuation
        
        # Enhanced preprocessing function
        def preprocess_query(text, is_chinese=False):
            # For non-Chinese text
            if not is_chinese:
                # Convert to lowercase
                text = text.lower()
                # Remove punctuation but preserve important symbols
                text = re.sub(f'[{re.escape(punctuation.replace("-", "").replace("_", ""))}]', ' ', text)
                # Remove extra whitespace
                text = re.sub(r'\s+', ' ', text).strip()
                
                # Extract key phrases (2-3 word combinations) to improve matching
                words = text.split()
                if len(words) > 1:
                    # Add bigrams and trigrams to the query
                    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
                    trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words)-2)] if len(words) > 2 else []
                    
                    # Combine original query with key phrases
                    enhanced_query = f"{text} {' '.join(bigrams)} {' '.join(trigrams)}"
                    return enhanced_query
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
            
            # Look for multi-character terms (enhanced approach)
            # In Chinese, meaningful terms are often 2-4 characters
            multi_char_terms = []
            
            # Add 2-character terms
            for i in range(len(key_terms) - 1):
                multi_char_terms.append(key_terms[i] + key_terms[i+1])
            
            # Add 3-character terms
            for i in range(len(key_terms) - 2):
                multi_char_terms.append(key_terms[i] + key_terms[i+1] + key_terms[i+2])
            
            # Add 4-character terms (very common in Chinese)
            for i in range(len(key_terms) - 3):
                multi_char_terms.append(key_terms[i] + key_terms[i+1] + key_terms[i+2] + key_terms[i+3])
                
            # Add the full query as a term for exact matching
            multi_char_terms.append(query)
            
            # Add sliding window segments of the query
            window_size = 10
            if len(query) > window_size:
                for i in range(len(query) - window_size + 1):
                    multi_char_terms.append(query[i:i+window_size])
            
            print(f"Generated {len(multi_char_terms)} multi-character terms and phrases")
            
            # Weight terms by frequency in the query
            term_weights = {}
            for term in multi_char_terms:
                if term in term_weights:
                    term_weights[term] += 1
                else:
                    term_weights[term] = 1
                    
            # Sort terms by length (longer terms are more specific)
            multi_char_terms.sort(key=lambda x: (term_weights[x], len(x)), reverse=True)
            
            # Keep the top terms to avoid query explosion
            multi_char_terms = multi_char_terms[:20]
            
        # We'll use the enhanced query processing similar to the original code
        # Apply preprocessing to query to get better retrieval
        processed_query = preprocess_query(query, is_chinese)
        
        # For Chinese queries, extract key terms to help with scoring
        if is_chinese:
            # Extract all Chinese characters as potential key terms
            key_terms = [char for char in query if '\u4e00' <= char <= '\u9fff']
            print(f"Extracted {len(key_terms)} Chinese characters")
            
            # Look for multi-character terms (enhanced approach)
            multi_char_terms = []
            
            # Add 2-character terms
            for i in range(len(key_terms) - 1):
                multi_char_terms.append(key_terms[i] + key_terms[i+1])
            
            # Add 3-character terms
            for i in range(len(key_terms) - 2):
                multi_char_terms.append(key_terms[i] + key_terms[i+1] + key_terms[i+2])
            
            # Add 4-character terms
            for i in range(len(key_terms) - 3):
                multi_char_terms.append(key_terms[i] + key_terms[i+1] + key_terms[i+2] + key_terms[i+3])
                
            # Add the full query as a term for exact matching
            multi_char_terms.append(query)
        
        # Next step is to deduplicate the results we already have
        update_progress(operation_id, current_step=8, 
                       description="Deduplicating results from different collections")
        
        # Use a dictionary for faster lookups during deduplication
        unique_docs = {}
        for doc in results:
            # Create a robust hash key using both content and metadata
            content_sample = doc.page_content[:150] if len(doc.page_content) > 150 else doc.page_content
            source = doc.metadata.get('source', '')
            source_collection = doc.metadata.get('source_collection', '')
            
            # Combine content, source and collection for a precise deduplication key
            dedup_key = f"{source}:{source_collection}:{content_sample}"
            
            # If we haven't seen this document before
            if dedup_key not in unique_docs:
                unique_docs[dedup_key] = doc
        
        # Convert back to list
        results = list(unique_docs.values())
        print(f"Deduplicated to {len(results)} unique documents")
        
        # If we have too many results, limit them
        if len(results) > n_results:
            results = results[:n_results]
        
        update_progress(operation_id, current_step=9, 
                      description=f"Finalized {len(results)} documents for context")
        
        # Last resort if we still have no results
        if not results and is_chinese and collections_to_search:
            print("Last resort: Searching for any Chinese content")
            # Try to find any documents with Chinese text in the first collection
            try:
                fallback_collection = collections_to_search[0]
                vectorstore = Chroma(
                    persist_directory=CHROMA_PERSIST_DIR, 
                    embedding_function=ollama_ef, 
                    collection_name=fallback_collection
                )
                
                # Try some common Chinese characters or terms from the query
                for char in key_terms[:10]:  # Try the first 10 characters
                    try:
                        char_results = vectorstore.similarity_search(char, k=5)
                        if char_results:
                            print(f"Found {len(char_results)} results for character '{char}'")
                            results.extend(char_results)
                            if len(results) >= n_results:
                                results = results[:n_results]
                                break
                    except Exception as e:
                        print(f"Error searching for character '{char}': {e}")
                        continue
            except Exception as e:
                print(f"Error in last resort search: {e}")
        
        print(f"Retrieved {len(results)} context chunks from knowledge base")
        
        # Format the context for the model with better organization and relevance scoring
        formatted_context = ""
        
        # Enhanced scoring based on multiple relevance indicators
        scored_results = []
        for i, doc in enumerate(results):
            # Extract source document name from metadata if available
            source = doc.metadata.get('source', 'Unknown source')
            # Get just the filename without path
            if source != 'Unknown source' and isinstance(source, str):
                source = os.path.basename(source)
            
            # Start with base score - use metadata score if available
            score = doc.metadata.get('score', 1.0)
            
            # Advanced content matching with term frequency and position weighting
            if is_chinese:
                # For Chinese, check for exact matches of multi-character terms
                if 'multi_char_terms' in locals() and multi_char_terms:
                    # Prioritize longer terms (more specific)
                    for i, term in enumerate(multi_char_terms[:10]):
                        # Weight by term length and position in the list
                        term_weight = (0.3 * len(term) / 10) * (1.0 - (i * 0.05))
                        
                        # Count occurrences for frequency weighting
                        term_count = doc.page_content.count(term)
                        if term_count > 0:
                            # Boost more for multiple occurrences
                            score += term_weight * min(term_count, 3)  # Cap at 3 occurrences
                            
                            # Extra boost for terms at the beginning of the content
                            if doc.page_content.find(term) < 100:
                                score += 0.2  # Beginning of content is often more relevant
                                
                    # Special boost for exact query match
                    if query in doc.page_content:
                        score += 1.0  # Major boost for exact query match
            else:
                # For non-Chinese, use more sophisticated word matching
                query_words = query.lower().split()
                
                # Count matching words and their positions
                matched_words = 0
                matched_at_beginning = 0
                content_lower = doc.page_content.lower()
                
                for word in query_words:
                    if len(word) > 3:  # Only consider meaningful words
                        # Count occurrences
                        word_count = content_lower.count(word)
                        if word_count > 0:
                            matched_words += 1
                            score += 0.1 * min(word_count, 5)  # Cap at 5 occurrences
                            
                            # Check if word appears at beginning
                            if content_lower.find(word) < 150:
                                matched_at_beginning += 1
                
                # Boost based on percentage of query words matched
                if len(query_words) > 0:
                    match_percentage = matched_words / len(query_words)
                    score += match_percentage * 0.5
                    
                # Boost for words appearing at beginning
                score += matched_at_beginning * 0.15
                
                # Check for exact phrase matches (much stronger signal)
                processed_query = ' '.join([w for w in query_words if len(w) > 3])
                if processed_query and processed_query in content_lower:
                    score += 1.5  # Major boost for exact phrase match
            
            # Content quality factors
            content_len = len(doc.page_content)
            
            # Penalize very short chunks (likely incomplete information)
            if content_len < 100:
                score -= 0.3
            # Slight penalty for extremely long chunks (harder to process)
            elif content_len > 5000:
                score -= 0.2
            # Prefer medium-length chunks (500-2000 chars)
            elif 500 <= content_len <= 2000:
                score += 0.2  # Bonus for ideal length
                
            # Boost documents with rich metadata (likely higher quality)
            if doc.metadata.get('title') and doc.metadata.get('title') != 'Unknown Title':
                score += 0.2
            if doc.metadata.get('author') and doc.metadata.get('author') != 'Unknown Author':
                score += 0.1
                
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
