import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from config import OLLAMA_HOST

# Import custom loaders
from custom_loaders import ChineseTheologyXMLLoader, ChineseTheologyXMLDirectoryLoader

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

ollama_ef = OllamaEmbeddings(
    base_url=OLLAMA_HOST,
    model=DEFAULT_EMBED_MODEL
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

def split_documents(documents):
    """Splits documents into smaller chunks with enhanced handling for Chinese text."""
    print("Splitting documents...")
    
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
    
    split_docs = text_splitter.split_documents(documents)
    print(f"Split into {len(split_docs)} chunks.")
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
        except Exception as e:
            print(f"Error processing document {filename}: {e}")
            return False
            
        print(f"Loading existing Chroma vector store from: {CHROMA_PERSIST_DIR}")
        # Load the existing vector store or create a new one
        try:
            vectorstore = Chroma(
                persist_directory=CHROMA_PERSIST_DIR, 
                embedding_function=ollama_ef, 
                collection_name=collection_name
            )
        except Exception as e:
            print(f"Error loading existing vector store: {e}")
            print("Creating a new vector store")
            vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=ollama_ef,
                persist_directory=CHROMA_PERSIST_DIR,
                collection_name=collection_name
            )
            vectorstore.persist()
            return True

        print(f"Adding {len(split_docs)} chunks from '{filename}' to the vector store.")
        # Add the new document chunks to the existing store
        vectorstore.add_documents(split_docs)
        
        print(f"Persisting updated vector store to: {CHROMA_PERSIST_DIR}")
        vectorstore.persist()
        print(f"--- Single document '{filename}' added successfully ---")
        return True

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

def retrieve_context(query, collection_name=DEFAULT_COLLECTION_NAME, conversation_id=None, n_results=500):
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
        # If conversation_id is provided, use a conversation-specific collection
        if conversation_id:
            collection_name = get_conversation_collection_name(conversation_id)
            
        print(f"Loading persisted Chroma vector store from: {CHROMA_PERSIST_DIR}, collection: {collection_name}")
        # Load the persisted vector store
        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR, 
            embedding_function=ollama_ef, 
            collection_name=collection_name
        )
        
        # Get total document count to determine retrieval strategy
        total_docs = vectorstore._collection.count()
        print(f"Total documents in collection: {total_docs}")
        
        # Determine how many documents to retrieve based on collection size
        # For all collections, try to retrieve a substantial portion of documents
        if total_docs < 5000:
            # For small collections, retrieve almost everything
            fetch_k = min(total_docs, 4000)  # Retrieve up to 4000 docs for small collections
            k_value = min(total_docs, n_results)
        elif total_docs < 20000:
            # For medium collections, retrieve a large portion
            fetch_k = min(int(total_docs * 0.7), 18000)  # Up to 70% of docs, max 18000
            k_value = min(int(total_docs * 0.3), n_results)  # Up to 30% of docs, max n_results
        else:
            # For very large collections, still retrieve a significant amount
            fetch_k = 18000  # Fixed large value
            k_value = n_results
        
        print(f"Retrieval strategy: fetch_k={fetch_k}, k={k_value}")
        
        # For Chinese text, we need to be more lenient with similarity scores
        # Use MMR retriever to maximize relevance and diversity
        retriever = vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance - balances relevance with diversity
            search_kwargs={
                "k": k_value,
                "fetch_k": fetch_k,  # Fetch more candidates to filter from
                "lambda_mult": 0.7  # Lower values (0.5-0.7) prioritize diversity over pure relevance
            }
        )
        
        # For Chinese queries, use specialized processing
        is_chinese = any('\u4e00' <= char <= '\u9fff' for char in query)
        
        print(f"Querying vector store for: '{query[:50]}...'")
        print(f"Query language detected: {'Chinese' if is_chinese else 'Other'}")
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
            
        # Use the retriever to get relevant documents with multiple strategies
        all_results = []
        try:
            # Strategy 1: Try original query first
            print("Strategy 1: Using original query")
            results1 = retriever.get_relevant_documents(query)
            print(f"Strategy 1 results: {len(results1)}")
            all_results.extend(results1)
            
            # For Chinese text, try additional strategies
            if is_chinese:
                # Strategy 2: Try without question marks and other punctuation
                clean_query = ''.join([c for c in query if '\u4e00' <= c <= '\u9fff' or c.isalnum()])
                if clean_query != query:
                    print(f"Strategy 2: Using cleaned query: '{clean_query}'")
                    results2 = retriever.get_relevant_documents(clean_query)
                    print(f"Strategy 2 results: {len(results2)}")
                    all_results.extend([r for r in results2 if r not in all_results])
                
                # Strategy 3: Try with key multi-character terms
                if multi_char_terms:
                    # Join the most important terms (first few multi-char terms)
                    term_query = ' '.join(multi_char_terms[:5])  
                    print(f"Strategy 3: Using key terms: '{term_query}'")
                    results3 = retriever.get_relevant_documents(term_query)
                    print(f"Strategy 3 results: {len(results3)}")
                    all_results.extend([r for r in results3 if r not in all_results])
                
                # Strategy 4: Direct similarity search as last resort
                if len(all_results) < 5:  # If we still don't have enough results
                    print("Strategy 4: Using direct similarity search")
                    results4 = vectorstore.similarity_search(query, k=n_results)
                    print(f"Strategy 4 results: {len(results4)}")
                    all_results.extend([r for r in results4 if r not in all_results])
            
            # Deduplicate and limit results
            seen = set()
            results = []
            for doc in all_results:
                # Use content hash as a simple deduplication key
                content_hash = hash(doc.page_content[:100])  # Use first 100 chars as hash key
                if content_hash not in seen:
                    seen.add(content_hash)
                    results.append(doc)
                if len(results) >= n_results:
                    break
                    
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
        else:
            print("No relevant context found in knowledge base")
            
        return formatted_context
    except Exception as e:
        print(f"Error retrieving context from Chroma vector store: {e}")
        print("Please ensure the vector store has been set up correctly.")
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
