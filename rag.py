import os
import re
import uuid
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import chromadb
from chromadb.config import Settings
from chromadb.errors import InvalidDimensionException
from web_search import search_web, format_web_results, get_web_context
from langchain_community.embeddings import OllamaEmbeddings
from config import DEFAULT_EMBEDDING_MODEL, USER_PREFERENCES
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from config import OLLAMA_HOST
import pandas as pd
import sys

# Import custom loaders
from custom_loaders import (
    ChineseTheologyXMLLoader, 
    ChineseTheologyXMLDirectoryLoader,
    CSVLoader,
    ExcelLoader,
    EnhancedExcelLoader
)

# Import hybrid search
from hybrid_search import hybrid_search, process_with_retry

# Import progress tracker
from progress_tracker import create_progress, update_progress, get_progress

# --- Configuration ---
# DEFAULT_EMBED_MODEL now imported from config.py
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "db", "chroma_db")
DEFAULT_COLLECTION_NAME = "rag_documents"
# Increased chunk size for better context retention in mixed-language documents
CHUNK_SIZE = 3000  # Increased from 2000 for better context in mixed-language content
CHUNK_OVERLAP = 600  # Increased overlap to maintain context between chunks (20% of CHUNK_SIZE)

# Function to sanitize keys for ChromaDB (remove invalid characters)
def sanitize_key(key):
    """
    Sanitize a key for use in ChromaDB by removing or replacing invalid characters.
    ChromaDB doesn't allow certain characters in collection names and keys.
    
    Args:
        key: The key string to sanitize
        
    Returns:
        str: Sanitized key string
    """
    if not key:
        return key
        
    # Replace colons, slashes, and other problematic characters
    sanitized = key.replace(':', '_').replace('/', '_').replace('\\', '_')
    
    # Remove any other non-alphanumeric characters except underscores and hyphens
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', sanitized)
    
    return sanitized

# Function to get conversation-specific collection name
def get_conversation_collection_name(conversation_id):
    """Generate a sanitized collection name for a specific conversation"""
    # Create the collection name and sanitize it
    collection_name = f"conversation_{conversation_id}"
    return sanitize_key(collection_name)

# Function to list all available collections
def list_collections():
    """List all available collections in the ChromaDB"""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        collections = client.list_collections()
        return [c.name for c in collections]
    except Exception as e:
        print(f"Error listing collections: {str(e)}")
        return []

# Ensure ChromaDB persistence directory exists
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# --- LangChain Embedding Function ---
# Use the model specified in user preferences (config.DEFAULT_EMBEDDING_MODEL)
print(f"Using embedding model from user preferences: {DEFAULT_EMBEDDING_MODEL}")

# Set up local file store for caching embeddings
store = LocalFileStore("./cache/embeddings")

# Adding secondary embeddings models for hybrid search approach
SECONDARY_EMBED_MODELS = [
    "mxbai-embed-large" if "mxbai-embed-large" in os.environ.get('AVAILABLE_EMBEDDING_MODELS', "") else None,
    "bge-large-zh" if "bge-large-zh" in os.environ.get('AVAILABLE_EMBEDDING_MODELS', "") else None
]
SECONDARY_EMBED_MODELS = [model for model in SECONDARY_EMBED_MODELS if model]

# Create base embeddings model using the user's preferred model
print(f"Initializing base embedding model: {DEFAULT_EMBEDDING_MODEL} from {OLLAMA_HOST}")
base_embeddings = OllamaEmbeddings(
    base_url=OLLAMA_HOST,
    model=DEFAULT_EMBEDDING_MODEL,
    show_progress=True
    # Note: OllamaEmbeddings doesn't support explicit dimensions parameter
    # The dimensions are determined by the model itself
)

# Create secondary embeddings if available
secondary_embeddings = []
for model in SECONDARY_EMBED_MODELS:
    try:
        print(f"Initializing secondary embedding model: {model}")
        emb = OllamaEmbeddings(
            base_url=OLLAMA_HOST,
            model=model,
            show_progress=True
            # Note: OllamaEmbeddings doesn't support explicit dimensions parameter
        )
        secondary_embeddings.append(emb)
        print(f"Added secondary embedding model: {model}")
    except Exception as e:
        print(f"Failed to load secondary embedding model {model}: {e}")

# Verify embedding dimensions
def verify_embedding_dimensions():
    """Verify embedding dimensions to ensure consistency"""
    try:
        print("Verifying embedding dimensions...")
        test_text = "This is a test sentence for embedding dimension verification."
        base_embedding = base_embeddings.embed_query(test_text)
        print(f"Base embedding model '{DEFAULT_EMBEDDING_MODEL}' dimensions: {len(base_embedding)}")
        
        # Check secondary embeddings if available
        for i, emb in enumerate(secondary_embeddings):
            sec_embedding = emb.embed_query(test_text)
            print(f"Secondary embedding model {i+1} dimensions: {len(sec_embedding)}")
            
        return True
    except Exception as e:
        print(f"Error verifying embedding dimensions: {e}")
        return False

# Run dimension verification
verify_embedding_dimensions()

# Function to check for embedding dimension mismatches in existing collections
def check_collection_dimensions(fix_mismatches=False):
    """Check if all collections have the same embedding dimensions as the current model.
    
    This function verifies that the dimensions of embeddings in existing collections
    match the dimensions produced by the current embedding model. Mismatches can occur
    when switching between different embedding models.
    
    Args:
        fix_mismatches: If True, attempt to fix dimension mismatches (not implemented yet)
        
    Returns:
        bool: True if all collections have matching dimensions, False otherwise
    """
    try:
        print("Checking embedding dimensions in existing collections...")
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        collections = client.list_collections()
        
        if not collections:
            print("No collections found in ChromaDB.")
            return True
            
        print(f"Found {len(collections)} collections in ChromaDB.")
        
        # Get base embedding dimensions
        test_text = "This is a test sentence for embedding dimension verification."
        base_embedding = base_embeddings.embed_query(test_text)
        base_dim = len(base_embedding)
        print(f"Current base embedding dimensions: {base_dim}")
        
        mismatched_collections = []
        
        # Check each collection
        for collection in collections:
            try:
                # Try to get collection metadata
                coll = client.get_collection(collection.name)
                if hasattr(coll, 'metadata') and coll.metadata:
                    if 'dimension' in coll.metadata:
                        coll_dim = coll.metadata['dimension']
                        print(f"Collection '{collection.name}' has dimension: {coll_dim}")
                        
                        if coll_dim != base_dim:
                            print(f"WARNING: Dimension mismatch in collection '{collection.name}'")
                            print(f"  Collection dimension: {coll_dim}, Current embedding dimension: {base_dim}")
                            mismatched_collections.append(collection.name)
                            
                            if fix_mismatches:
                                print(f"  Attempting to fix by re-indexing collection '{collection.name}'...")
                                # This would require re-indexing the collection
                                # For now, just mark it for manual re-indexing
                                print(f"  Manual re-indexing required for collection '{collection.name}'")
                    else:
                        print(f"Collection '{collection.name}' has no dimension metadata.")
            except Exception as e:
                print(f"Error checking collection '{collection.name}': {e}")
        
        if mismatched_collections:
            print(f"\nFound {len(mismatched_collections)} collections with dimension mismatches:")
            for coll_name in mismatched_collections:
                print(f"  - {coll_name}")
            print("\nTo fix these collections, you need to re-index them with the current embedding model.")
            print("This can be done by exporting the documents, deleting the collections, and re-importing.")
        else:
            print("\nAll collections have matching dimensions with the current embedding model.")
                
        return True
    except Exception as e:
        print(f"Error checking collection dimensions: {e}")
        return False

# Run collection dimension check
check_collection_dimensions()

# Create cached embeddings to improve performance
# Sanitize the model name for use in ChromaDB namespace
sanitized_model_name = sanitize_key(base_embeddings.model)
print(f"Using sanitized model name for embeddings cache: {sanitized_model_name}")

ollama_ef = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=base_embeddings,
    document_embedding_cache=store,
    namespace=sanitized_model_name
)

# --- Document Loading and Processing ---

def add_documents(documents, collection_name=DEFAULT_COLLECTION_NAME, conversation_id=None):
    """Add documents to the vector store.
    
    Args:
        documents: List of Document objects to add
        collection_name: Name of the collection to add to
        conversation_id: If provided, will use a conversation-specific collection
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Use conversation-specific collection if ID provided
        if conversation_id:
            collection_name = get_conversation_collection_name(conversation_id)
            
        print(f"Adding {len(documents)} documents to collection: {collection_name}")
        
        # Create or get the Chroma collection
        chroma_db = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name=collection_name,
            embedding_function=ollama_ef
        )
        
        # Add documents to the collection
        chroma_db.add_documents(documents)
        
        # Persist the changes
        chroma_db.persist()
        print(f"Successfully added {len(documents)} documents to {collection_name}")
        return True
        
    except Exception as e:
        print(f"Error adding documents to vector store: {e}")
        return False

def load_documents(source_directory):
    """Loads documents from various file types in the specified directory."""
    print(f"Loading documents from: {source_directory}")
    documents = []
    
    # Text files
    loader_txt = DirectoryLoader(source_directory, glob="**/*.txt", loader_cls=TextLoader, recursive=True, show_progress=True)
    documents.extend(loader_txt.load())
    
    # PDF files
    loader_pdf = DirectoryLoader(source_directory, glob="**/*.pdf", loader_cls=PyPDFLoader, recursive=True, show_progress=True)
    documents.extend(loader_pdf.load())

    # Markdown files
    loader_md = DirectoryLoader(source_directory, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader, recursive=True, show_progress=True)
    documents.extend(loader_md.load())
    
    # CSV files with enhanced debug output
    try:
        print("Loading CSV files with enhanced loader...")
        csv_files = [f for f in Path(source_directory).rglob('*.csv') if f.is_file()]
        
        for csv_file in csv_files:
            try:
                print(f"\nProcessing CSV file: {csv_file.name}")
                
                # Use CSVLoader with more detailed configuration
                loader = CSVLoader(
                    str(csv_file),
                    csv_args={
                        'encoding': 'utf-8',
                        'on_bad_lines': 'warn',
                        'low_memory': False
                    },
                    metadata_columns=[]  # Add any columns you want in metadata
                )
                
                # Load documents
                csv_docs = loader.load()
                print(f"Loaded {len(csv_docs)} documents from {csv_file.name}")
                
                # Debug: Print first few documents' content
                for i, doc in enumerate(csv_docs[:3]):  # Show first 3 documents
                    preview = doc.page_content[:300]  # Show first 300 chars
                    if len(doc.page_content) > 300:
                        preview += "..."
                    print(f"Document {i+1} preview: {preview}")
                    
                    # Print metadata keys for inspection
                    if i == 0:  # Only show for first doc to avoid too much output
                        print(f"Metadata keys: {list(doc.metadata.keys())}")
                
                documents.extend(csv_docs)
                
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                import traceback
                traceback.print_exc()
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        import traceback
        traceback.print_exc()
    
    # Excel files (.xls, .xlsx) - Using EnhancedExcelLoader
    try:
        print("Loading Excel files with enhanced loader...")
        excel_files = []
        excel_files.extend(Path(source_directory).rglob('*.xls'))
        excel_files.extend(Path(source_directory).rglob('*.xlsx'))
        
        for excel_file in excel_files:
            try:
                print(f"Processing Excel file: {excel_file.name}")
                # Use EnhancedExcelLoader with more detailed logging
                loader = EnhancedExcelLoader(
                    str(excel_file),
                    detect_merged_cells=True,
                    detect_multi_level_headers=True,
                    detect_tables=True,
                    verbose=True
                )
                excel_docs = loader.load()
                print(f"Loaded {len(excel_docs)} documents from {excel_file.name}")
                
                # Debug: Print first few documents' content
                for i, doc in enumerate(excel_docs[:3]):
                    print(f"Document {i+1} preview: {doc.page_content[:200]}..." if len(doc.page_content) > 200 else f"Document {i+1}: {doc.page_content}")
                
                documents.extend(excel_docs)
            except Exception as e:
                print(f"Error processing {excel_file}: {e}")
                import traceback
                traceback.print_exc()
    except Exception as e:
        print(f"Error loading Excel files: {e}")
        import traceback
        traceback.print_exc()
    
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

    print(f"Loaded {len(documents)} documents in total.")
    return documents

def split_documents(documents, operation_id=None):
    """Splits documents into smaller chunks with enhanced handling for mixed-language text.
    Uses parallel processing for large document sets and optimizes for both English and Chinese content."""
    print("Splitting documents...")
    if operation_id:
        update_progress(operation_id, description="Analyzing documents for optimal splitting")
    
    # Check language composition in documents
    documents_with_chinese = 0
    total_chinese_chars = 0
    total_english_chars = 0
    total_chars = 0
    
    for doc in documents:
        content = doc.page_content
        total_chars += len(content)
        chinese_chars = sum(1 for char in content if '\u4e00' <= char <= '\u9fff')
        # Count English characters (approximation)
        english_chars = sum(1 for char in content if ('a' <= char.lower() <= 'z') or char.isspace())
        
        total_chinese_chars += chinese_chars
        total_english_chars += english_chars
        
        if chinese_chars > 0:
            documents_with_chinese += 1
    
    # Calculate percentage of Chinese and English text
    chinese_percentage = total_chinese_chars / max(1, total_chars) * 100
    english_percentage = total_english_chars / max(1, total_chars) * 100
    
    print(f"Language analysis: {chinese_percentage:.1f}% Chinese, {english_percentage:.1f}% English/Latin characters")
    
    # Create a balanced splitter that works well for both languages
    if documents_with_chinese > 0:
        print(f"Mixed language content detected - {documents_with_chinese}/{len(documents)} documents contain Chinese ({chinese_percentage:.1f}% Chinese characters)")
        print("Using enhanced mixed-language splitter with improved English preservation")
        
        # Enhanced splitter with better English content preservation
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=[
                # Major section breaks (double newlines)
                "\n\n", 
                # Single newlines
                "\n",
                # Paragraph and section markers
                "## ", "### ", "#### ",
                # Sentence boundaries (both languages) - note the space after punctuation for English
                ". ", "。", "! ", "！", "? ", "？",
                # English sentence boundaries without space (end of line)
                ".", "!", "?",
                # Clause boundaries (both languages)
                "; ", "；", ", ", "，", 
                # Table and list markers
                "|", "- ", "* ", "1. ", "2. ",
                # Word boundaries (space and no space as last resort)
                " ", ""
            ],
            keep_separator=True,
            # Ensure we don't split in the middle of words or CJK characters
            is_separator_regex=False
        )
        
        # Log the splitting configuration
        print(f"Using enhanced mixed-language text splitter with chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    else:
        print("English-only content detected - using enhanced English splitter")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP,
            separators=[
                # Major section breaks
                "\n\n", "\n",
                # Headers and section markers
                "## ", "### ", "#### ",
                # Sentence boundaries with space
                ". ", "! ", "? ",
                # Sentence boundaries without space (end of line)
                ".", "!", "?",
                # Clause boundaries
                "; ", ", ",
                # Table and list markers
                "|", "- ", "* ", "1. ", "2. ",
                # Word boundaries
                " ", ""
            ],
            keep_separator=True,
            is_separator_regex=False
        )
        
        print(f"Using enhanced English text splitter with chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    
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
        # Debug information
        print(f"DEBUG: Received file_path: '{file_path}'")
        print(f"DEBUG: Current working directory: '{os.getcwd()}'")
        
        # If conversation_id is provided, use a conversation-specific collection
        if conversation_id:
            collection_name = get_conversation_collection_name(conversation_id)
            print(f"DEBUG: Using collection: '{collection_name}'")
        
        # Validate file path
        original_path = file_path
        if not os.path.isabs(file_path):
            # Try to convert relative path to absolute
            abs_path = os.path.abspath(file_path)
            print(f"DEBUG: Converted relative path to absolute: '{abs_path}'")
            if os.path.exists(abs_path):
                file_path = abs_path
                print(f"DEBUG: Using absolute path: '{file_path}'")
            else:
                print(f"DEBUG: Absolute path does not exist: '{abs_path}'")
        
        # Implement retry mechanism with delay for file existence check
        max_retries = 5
        retry_delay = 1  # seconds
        file_found = False
        file_path_to_use = None
        
        for attempt in range(max_retries):
            # Check if file exists
            file_exists = os.path.exists(file_path)
            print(f"DEBUG: File exists check (attempt {attempt+1}/{max_retries}): {file_exists} for path '{file_path}'")
            
            # Try alternative path formats
            alt_path = file_path.replace('\\', '/')
            alt_exists = os.path.exists(alt_path)
            print(f"DEBUG: Alternative path exists check: {alt_exists} for path '{alt_path}'")
            
            # Try checking if the file exists in the docs directory
            docs_path = os.path.join(os.path.dirname(__file__), "docs", os.path.basename(file_path))
            docs_exists = os.path.exists(docs_path)
            print(f"DEBUG: Docs directory path exists check: {docs_exists} for path '{docs_path}'")
            
            # Check if file exists in conversation-specific directory
            if conversation_id:
                conv_path = os.path.join(os.path.dirname(__file__), "docs", f"conversation_{conversation_id}", os.path.basename(file_path))
                conv_exists = os.path.exists(conv_path)
                print(f"DEBUG: Conversation path exists check: {conv_exists} for path '{conv_path}'")
            else:
                conv_path = None
                conv_exists = False
            
            # Determine which path to use
            if file_exists:
                file_path_to_use = file_path
                file_found = True
                break
            elif alt_exists:
                file_path_to_use = alt_path
                file_found = True
                break
            elif docs_exists:
                file_path_to_use = docs_path
                file_found = True
                break
            elif conv_exists:
                file_path_to_use = conv_path
                file_found = True
                break
            else:
                print(f"DEBUG: File not found on attempt {attempt+1}/{max_retries}, waiting {retry_delay} seconds...")
                time.sleep(retry_delay)
                # Increase delay for next attempt (exponential backoff)
                retry_delay *= 1.5
        
        if not file_found:
            error_msg = f"Error loading file: [Errno 2] No such file or directory: '{original_path}' after {max_retries} attempts"
            print(error_msg)
            return False
        
        file_path = file_path_to_use
        print(f"DEBUG: Using file path: '{file_path}'")
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
            elif file_extension == '.csv':
                # Use our custom CSV loader
                loader = CSVLoader(file_path)
                print(f"Using custom CSV loader for {filename}")
            elif file_extension in ['.xls', '.xlsx']:
                # Use our enhanced Excel loader with advanced features
                loader = EnhancedExcelLoader(
                    file_path,
                    detect_merged_cells=True,
                    detect_multi_level_headers=True,
                    detect_tables=True,
                    extract_formulas=True,
                    verbose=True
                )
                print(f"Using enhanced Excel loader for {filename}")
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

def retrieve_context(query, collection_name=DEFAULT_COLLECTION_NAME, conversation_id=None, bot_collection=None, n_results=2000, operation_id=None, recursive_depth=0, max_recursive_depth=1, web_search=False):
    """Retrieves relevant document chunks for a given query using LangChain Chroma.
    
    Args:
        query: The query to search for
        collection_name: Name of the collection to search in (default collection)
        conversation_id: If provided, will search in a conversation-specific collection
        bot_collection: If provided, will search in the bot's knowledge base collection
        n_results: Number of results to return
        operation_id: Unique identifier for tracking the retrieval process
        recursive_depth: Current depth of recursive search
        max_recursive_depth: Maximum depth for recursive search
        web_search: If True, perform a web search instead of using local documents
        
    Returns:
        str: The retrieved context as a string
    """
    # Handle web search if enabled
    if web_search and recursive_depth == 0:
        # Only perform web search at the top level
        web_context = get_web_context(query)
        if web_context:
            return f"{web_context}\n\n"
            
    # Clean up query if web search was in the query string
    if '--web' in query:
        query = re.sub(r'\s*--web\b', '', query).strip()
    
    try:
        # Initialize Chroma client
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        
        # Generate a unique operation ID if not provided
        if operation_id is None:
            operation_id = f"rag_{uuid.uuid4().hex[:8]}"
            
        # Check if query is in Chinese - do this first so we have it for all processing
        chinese_char_count = sum(1 for char in query if '\u4e00' <= char <= '\u9fff')
        total_chars = len(query.strip())
        chinese_percentage = chinese_char_count / total_chars if total_chars > 0 else 0
        
        # Consider mixed language if less than 80% Chinese
        is_chinese = chinese_percentage > 0.2  # Even small amount of Chinese triggers Chinese processing
        is_mixed = 0 < chinese_percentage < 0.8  # Mixed language handling
        
        print(f"!!! QUERY LANGUAGE DETECTION: {'MIXED' if is_mixed else 'CHINESE' if is_chinese else 'ENGLISH'} !!!")
        print(f"!!! QUERY: '{query[:50]}...' !!!")
        print(f"!!! CHINESE PERCENTAGE: {chinese_percentage:.1%} ({chinese_char_count}/{total_chars} chars) !!!")
        
        # Define query preprocessing function here so it's available immediately
        def preprocess_query(query_text, *args):
            """Preprocess the query for better retrieval results.
            
            Args:
                query_text: The query string to process
                *args: Additional arguments that might be passed by the retriever
                
            Returns:
                str: Enhanced query string for better retrieval
            """
            # Convert to lowercase for consistency
            text = query_text.lower()
            
            # Log if we're getting extra arguments (for debugging)
            if args:
                print(f"Note: preprocess_query received {len(args)} extra arguments")
                # We'll ignore extra arguments but still process the query_text
            
            # Extract Chinese characters for processing
            chinese_chars = [c for c in text if '\u4e00' <= c <= '\u9fff']
            
            # Special handling for Chinese or mixed-language text
            if is_chinese:
                # Remove Chinese punctuation but preserve the core characters
                text = re.sub(r'[，。？！：；''""（）【】《》、～]', ' ', text)
                
                # Create 2-character and 3-character combinations
                char_bigrams = [chinese_chars[i] + chinese_chars[i+1] for i in range(len(chinese_chars)-1)] if len(chinese_chars) > 1 else []
                char_trigrams = [chinese_chars[i] + chinese_chars[i+1] + chinese_chars[i+2] for i in range(len(chinese_chars)-2)] if len(chinese_chars) > 2 else []
                
                # For longer Chinese text, also add some 4-character combinations (common in Chinese)
                char_quadgrams = [chinese_chars[i] + chinese_chars[i+1] + chinese_chars[i+2] + chinese_chars[i+3] for i in range(len(chinese_chars)-3)] if len(chinese_chars) > 3 else []
                
                # Combine all to enhance search matching capability
                chinese_enhanced_terms = chinese_chars + char_bigrams + char_trigrams + char_quadgrams
                
                # Process English parts if mixed language
                if is_mixed:
                    # Extract non-Chinese text (likely English)
                    english_text = ''.join([c for c in text if not ('\u4e00' <= c <= '\u9fff')])
                    
                    # Clean English text
                    english_text = re.sub(f'[{re.escape(punctuation.replace("-", "").replace("_", ""))}]', ' ', english_text)
                    english_text = re.sub(r'\s+', ' ', english_text).strip()
                    
                    # Extract key phrases from English
                    words = english_text.split()
                    english_enhanced_terms = []
                    
                    # Add individual words first
                    english_enhanced_terms.extend(words)
                    
                    # Add word n-grams if we have multiple words
                    if len(words) > 1:
                        # Add bigrams
                        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
                        english_enhanced_terms.extend(bigrams)
                        
                        # Add trigrams if we have enough words
                        if len(words) > 2:
                            trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words)-2)]
                            english_enhanced_terms.extend(trigrams)
                    
                    # Combine both language enhancements with higher weight for English terms
                    # Triple the English terms to ensure they're not overshadowed by Chinese
                    all_terms = chinese_enhanced_terms + english_enhanced_terms * 3
                    
                    # Add original query first for exact matching
                    enhanced_query = text + ' ' + ' '.join(all_terms)
                    
                    print(f"Enhanced mixed-language query with {len(chinese_enhanced_terms)} Chinese terms and {len(english_enhanced_terms)} English terms (tripled)")
                    return enhanced_query
                else:
                    # Chinese-only processing
                    enhanced_query = text + ' ' + ' '.join(chinese_enhanced_terms)
                    print(f"Enhanced Chinese query with {len(chinese_enhanced_terms)} additional terms")
                    return enhanced_query
            # For non-Chinese text
            else:
                # English-only processing
                # Remove punctuation but preserve hyphens and underscores
                text = re.sub(f'[{re.escape(punctuation.replace("-", "").replace("_", ""))}]', ' ', text)
                
                # Normalize whitespace
                text = re.sub(r'\s+', ' ', text).strip()
                
                # For English, add word n-grams to enhance matching
                words = text.split()
                enhanced_terms = []
                
                # Add individual words first
                enhanced_terms.extend(words)
                
                if len(words) > 1:
                    # Add bigrams to the query
                    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
                    enhanced_terms.extend(bigrams)
                    
                    # Add trigrams if we have enough words
                    if len(words) > 2:
                        trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words)-2)]
                        enhanced_terms.extend(trigrams)
                    
                    # Combine original text with n-grams
                    enhanced_query = text + ' ' + ' '.join(enhanced_terms)
                    print(f"Enhanced English query with {len(enhanced_terms)} additional terms")
                    return enhanced_query
            return text
            
        # Process the query right away to make sure it's available throughout the function
        # Import requirements for preprocessing
        import re
        from string import punctuation
        
        # Apply preprocessing based on language
        # Call preprocess_query with the query text
        processed_query = preprocess_query(query)
        print(f"Original query: '{query[:50]}...'")
        print(f"Processed query: '{processed_query[:50]}...'")
        print(f"Processed length: {len(processed_query)} chars")
        
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
        
        # Check if no specific collections were found, don't use ANY collection
        # This enforces our rule that RAG should only use specific collections based on our rules
        if not collections_to_search:
            print(f"No valid collections found based on our rules, not falling back to default")
            # Return empty results immediately to avoid any RAG search
            return ""
        
        # FOR CHINESE QUERIES: Immediately fetch samples from the collection if it's a Chinese query
        # This ensures we get at least something for Chinese queries
        if is_chinese and collections_to_search and 'bot_' in collections_to_search[0]:
            print(f"*** CHINESE QUERY DETECTED: IMMEDIATELY FETCHING SAMPLES FROM {collections_to_search[0]} ***")
            try:
                # Get the underlying collection directly
                if client.list_collections():
                    for coll in client.list_collections():
                        if coll.name == collections_to_search[0]:
                            # Peek some documents
                            try:
                                                # Get larger random sample of documents for deeper context
                                random_docs = coll.peek(limit=50)  # Increased from 20 to 50
                                
                                if random_docs and 'documents' in random_docs and len(random_docs['documents']) > 0:
                                    print(f"FOUND {len(random_docs['documents'])} SAMPLE DOCUMENTS")
                                    
                                    # Convert to string for context
                                    context_docs = []
                                    for i, doc in enumerate(random_docs['documents']):
                                        if i < 15:  # Increased from 5 to 15 samples for deeper context
                                            context_docs.append(f"Document {i+1}:\n{doc}")
                                    
                                    if context_docs:
                                        forced_context = "\n\n".join(context_docs)
                                        print(f"*** FORCED CHINESE CONTEXT: {len(forced_context)} chars ***")
                                        return forced_context
                            except Exception as e:
                                print(f"Error peeking documents: {e}")
            except Exception as e:
                print(f"Error in forced Chinese sample retrieval: {e}")
            
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
        def process_collection(collection_name):
            """Process a single collection and return relevant documents.
            
            Args:
                collection_name: Name of the collection to process
                
            Returns:
                List of Document objects from the collection
            """
            print(f"Processing collection {collection_name} for {'CHINESE' if is_chinese else 'NON-CHINESE'} query")
            try:
                # Connect to the collection
                vectorstore = Chroma(
                    persist_directory=CHROMA_PERSIST_DIR, 
                    embedding_function=ollama_ef, 
                    collection_name=collection_name
                )
                
                # Get document count
                docs_in_collection = 0
                try:
                    # Get collection stats
                    docs_in_collection = client.get_collection(collection_name).count()
                    print(f"Collection {collection_name} has {docs_in_collection} documents")
                    
                    # For truly maximum depth, when collections are under a threshold,
                    # just read the entire collection directly
                    if docs_in_collection > 0 and docs_in_collection <= 100:
                        print(f"SMALL COLLECTION DETECTED ({docs_in_collection} docs) - READING ENTIRE COLLECTION")
                        try:
                            # For small collections, just get everything
                            all_docs = client.get_collection(collection_name).peek(limit=docs_in_collection)
                            if all_docs and 'documents' in all_docs and len(all_docs['documents']) > 0:
                                # Convert to Document objects to match expected return type
                                from langchain.schema import Document
                                results = []
                                for i, doc_text in enumerate(all_docs['documents']):
                                    metadata = all_docs.get('metadatas', [{}] * len(all_docs['documents']))[i] or {}
                                    if 'source_collection' not in metadata:
                                        metadata['source_collection'] = collection_name
                                    results.append(Document(page_content=doc_text, metadata=metadata))
                                
                                print(f"Using full collection: {len(results)} documents")
                                return results
                        except Exception as full_e:
                            print(f"Failed to get full collection: {full_e}")
                            # Continue with normal retrieval
                    
                    # Use the hybrid search function for retrieval
                    return hybrid_search(
                        query=processed_query,
                        collection_name=collection_name,
                        embedding_functions=[ollama_ef],
                        k=n_results,
                        fetch_k=min(docs_in_collection * 2, 2000),
                        threshold=0.01 if is_chinese else 0.05,
                        is_chinese=is_chinese
                    )
                except Exception as e:
                    print(f"Error getting collection stats: {e}")
                    return []
            except Exception as e:
                print(f"Error connecting to collection {collection_name}: {e}")
                return []
        
        # Function for hybrid search combining multiple embedding approaches
        def hybrid_search(query, collection_name, embedding_functions, k=10, fetch_k=50, threshold=0.5, is_chinese=False):
            """Enhanced hybrid search that works for both Chinese and English content.
            
            Args:
                query: The query string
                collection_name: Name of the collection to search
                embedding_functions: List of embedding functions to use
                k: Number of results to return
                fetch_k: Number of candidates to fetch before filtering
                threshold: Score threshold for filtering results
                is_chinese: Whether the query is in Chinese (affects scoring strategy)
            """
            print(f"Processing collection {collection_name} for {'CHINESE' if is_chinese else 'NON-CHINESE'} query")
            try:
                print(f"Processing collection: {collection_name}")
                # Connect to the collection
                vectorstore = Chroma(
                    persist_directory=CHROMA_PERSIST_DIR, 
                    embedding_function=ollama_ef, 
                    collection_name=collection_name
                )
                
                # Get document count
                docs_in_collection = 0
                try:
                    # Get collection stats
                    docs_in_collection = client.get_collection(collection_name).count()
                    print(f"Collection {collection_name} has {docs_in_collection} documents")
                    
                    # For truly maximum depth, when collections are under a threshold,
                    # just read the entire collection directly
                    if docs_in_collection > 0 and docs_in_collection <= 100:
                        print(f"SMALL COLLECTION DETECTED ({docs_in_collection} docs) - READING ENTIRE COLLECTION")
                        try:
                            # For small collections, just get everything
                            all_docs = client.get_collection(collection_name).peek(limit=docs_in_collection)
                            if all_docs and 'documents' in all_docs and len(all_docs['documents']) > 0:
                                # Convert to Document objects to match expected return type
                                from langchain.schema import Document
                                results = []
                                for i, doc_text in enumerate(all_docs['documents']):
                                    metadata = all_docs.get('metadatas', [{}] * len(all_docs['documents']))[i] or {}
                                    if 'source_collection' not in metadata:
                                        metadata['source_collection'] = collection_name
                                    results.append(Document(page_content=doc_text, metadata=metadata))
                                
                                print(f"Using full collection: {len(results)} documents")
                                return results
                        except Exception as full_e:
                            print(f"Failed to get full collection: {full_e}")
                            # Continue with normal retrieval
                    
                    search_percentage = 0.5  # Increase default search percentage of collection
                    
                    # Adjust based on collection size
                    if docs_in_collection < 10:
                        search_percentage = 1.0  # For tiny collections, use all docs
                    elif docs_in_collection < 50:
                        search_percentage = 0.8  # For small collections, use most docs
                    elif docs_in_collection < 200:
                        search_percentage = 0.6  # For medium collections
                    
                    # Use balanced retrieval strategies for all languages
                    # Ensure we retrieve enough documents regardless of language
                    search_percentage = max(search_percentage, 0.9)  # Increase to 90% of documents for all queries
                    k_value = min(int(docs_in_collection * search_percentage), n_results)
                    fetch_k = min(docs_in_collection, max(k_value * 4, 2000))  # Fetch more candidates for all languages
                    
                    # Adjust retrieval strategy based on language detection
                    if is_mixed:
                        # For mixed-language queries, use even more aggressive parameters
                        fetch_k = min(docs_in_collection, max(k_value * 8, 4000))  # Fetch even more candidates for mixed language
                        print(f"Using MIXED-LANGUAGE retrieval strategy for {collection_name}: fetch_k={fetch_k}, k={k_value}")
                    elif is_chinese:
                        # For Chinese queries, use standard aggressive parameters
                        fetch_k = min(docs_in_collection, max(k_value * 6, 3000))
                        print(f"Using CHINESE retrieval strategy for {collection_name}: fetch_k={fetch_k}, k={k_value}")
                    else:
                        # For English queries, use standard aggressive parameters
                        fetch_k = min(docs_in_collection, max(k_value * 6, 3000))
                        print(f"Using ENGLISH retrieval strategy for {collection_name}: fetch_k={fetch_k}, k={k_value}")
                        
                    # Execute search based on language
                    try:
                        # Use hybrid search for all languages with appropriate parameters
                        # Use a very low threshold for all content to ensure we don't miss any segments
                        threshold_value = 0.01  # Use same low threshold for all languages to avoid bias
                        
                        # Log the search strategy
                        if is_mixed:
                            print(f"*** EXECUTING MIXED-LANGUAGE HYBRID SEARCH: '{processed_query[:50]}...' ***")
                        elif is_chinese:
                            print(f"*** EXECUTING CHINESE-OPTIMIZED HYBRID SEARCH: '{processed_query[:50]}...' ***")
                        else:
                            print(f"*** EXECUTING ENGLISH-OPTIMIZED HYBRID SEARCH: '{processed_query[:50]}...' ***")
                            
                        # Execute hybrid search with multiple embedding functions
                        results = []
                        all_candidates = []
                        
                        # Use each embedding function to get candidates
                        for i, emb_func in enumerate(embedding_functions):
                            try:
                                # Use the current embedding function
                                vectorstore.embedding_function = emb_func
                                
                                # Create retriever with appropriate parameters
                                retriever = vectorstore.as_retriever(
                                    search_type="similarity_score_threshold",  # Use threshold-based retrieval
                                    search_kwargs={
                                        "k": k_value,
                                        "fetch_k": fetch_k,
                                        "score_threshold": threshold_value  # Apply consistent threshold
                                    }
                                )
                                
                                # Get documents using this embedding
                                try:
                                    emb_results = retriever.get_relevant_documents(query)
                                    print(f"Embedding {i+1}/{len(embedding_functions)} retrieved {len(emb_results)} documents")
                                except Exception as retry_e:
                                    print(f"First attempt failed: {retry_e}, retrying...")
                                    # Simple retry logic
                                    time.sleep(1)
                                    try:
                                        emb_results = retriever.get_relevant_documents(query)
                                        print(f"Retry successful: {len(emb_results)} documents retrieved")
                                    except Exception as final_e:
                                        print(f"Retry failed: {final_e}")
                                        emb_results = []
                                
                                # Add to candidates
                                all_candidates.extend(emb_results)
                            except Exception as emb_e:
                                print(f"Error with embedding {i+1}: {emb_e}")
                        
                        # Deduplicate results by content
                        seen_contents = set()
                        for doc in all_candidates:
                            if doc.page_content not in seen_contents:
                                seen_contents.add(doc.page_content)
                                results.append(doc)
                        
                        # Sort by relevance (approximate)
                        results = results[:k_value]
                        print(f"Hybrid search retrieved {len(results)} documents from {collection_name}")
                    except Exception as e:
                        print(f"Error in hybrid search: {e}")
                        # Fall back to appropriate retrieval method
                        try:
                            if is_chinese:
                                # For Chinese, use similarity search with low threshold
                                print(f"Falling back to similarity search with fetch_k={fetch_k}")
                                retriever = vectorstore.as_retriever(
                                    search_type="similarity",
                                    search_kwargs={
                                        "k": k_value,
                                        "fetch_k": fetch_k,
                                        "score_threshold": 0.01  # Low threshold for Chinese
                                    }
                                )
                            else:
                                # For English, use MMR with balanced parameters
                                print(f"Falling back to MMR search with fetch_k={fetch_k}")
                                retriever = vectorstore.as_retriever(
                                    search_type="mmr",
                                    search_kwargs={
                                        "k": k_value,
                                        "fetch_k": fetch_k,
                                        "lambda_mult": 0.7,
                                        "score_threshold": 0.05  # Threshold for English
                                    }
                                )
                                
                            # Execute the fallback retrieval
                            print(f"*** EXECUTING FALLBACK SEARCH FOR: '{processed_query[:50]}...' ***")
                            try:
                                results = retriever.get_relevant_documents(query)
                                print(f"Retrieved {len(results)} documents from {collection_name}")
                            except Exception as retry_e:
                                print(f"First fallback attempt failed: {retry_e}, retrying...")
                                # Simple retry logic
                                time.sleep(1)
                                try:
                                    results = retriever.get_relevant_documents(query)
                                    print(f"Fallback retry successful: {len(results)} documents retrieved")
                                except Exception as final_e:
                                    print(f"Fallback retry failed: {final_e}")
                                    results = []
                        except Exception as e2:
                            print(f"Error in fallback retrieval: {e2}")
                            results = []
                    
                    # RECURSIVE RETRIEVAL - for maximal depth, do a second-level search
                    # on terms found in the top results if we haven't already recursed too deep
                    if recursive_depth < max_recursive_depth and len(results) > 0 and is_chinese:
                        try:
                            print(f"Starting recursive retrieval at depth {recursive_depth+1}")
                            # Extract key terms from top results
                            top_results_text = "".join([doc.page_content for doc in results[:5]])
                            
                            # Get new terms that weren't in original query
                            new_terms = set()
                            
                            # For Chinese, extract multi-character terms
                            if is_chinese and len(top_results_text) > 0:
                                # Find all 2-3 character sequences that appear multiple times
                                for i in range(len(top_results_text) - 2):
                                    term2 = top_results_text[i:i+2]
                                    if len(term2.strip()) == 2 and top_results_text.count(term2) > 1 and term2 not in query:
                                        new_terms.add(term2)
                                        
                                    if i < len(top_results_text) - 3:
                                        term3 = top_results_text[i:i+3]
                                        if len(term3.strip()) == 3 and top_results_text.count(term3) > 1 and term3 not in query:
                                            new_terms.add(term3)
                            
                            # Use the top 3 new terms for recursive search
                            new_terms_list = list(new_terms)[:3]
                            if new_terms_list:
                                print(f"Found {len(new_terms_list)} new terms for recursive search: {new_terms_list}")
                                
                                # For each new term, do a retrieval and add results
                                for term in new_terms_list:
                                    # Recursive call with increased depth counter
                                    term_results = retrieve_context(
                                        query=term, 
                                        collection_name=collection_name,
                                        conversation_id=conversation_id,
                                        bot_collection=bot_collection,
                                        n_results=100,  # Limit recursive search
                                        operation_id=operation_id,
                                        recursive_depth=recursive_depth+1,
                                        max_recursive_depth=max_recursive_depth
                                    )
                                    
                                    # Convert string results back to documents for merging
                                    if term_results:
                                        # Create a document from the string results
                                        from langchain.schema import Document
                                        term_doc = Document(
                                            page_content=term_results,
                                            metadata={"source": f"recursive_search_{term}", "score": 0.7}
                                        )
                                        results.append(term_doc)
                        except Exception as rec_e:
                            print(f"Error in recursive retrieval: {rec_e}")
                            # Continue with existing results {collection_name} using retriever")
                    
                    # If Chinese query and no results, try direct similarity search
                    if is_chinese and len(results) == 0:
                        print("*** CHINESE QUERY WITH NO RESULTS - TRYING DIRECT SIMILARITY SEARCH ***")
                        # Try direct similarity search with very low threshold
                        results = vectorstore.similarity_search(query, k=50, score_threshold=0.01)
                        print(f"Direct similarity search retrieved {len(results)} documents")
                except Exception as e:
                    print(f"Error in retrieval: {e}")
                    # If error, try basic similarity as backup
                    try:
                        results = vectorstore.similarity_search(query, k=50)
                        print(f"Fallback similarity search retrieved {len(results)} documents")
                    except Exception as e2:
                        print(f"Error in fallback retrieval: {e2}")
                        results = []
                
                print(f"Final result count from collection {collection_name}: {len(results)}")
                
                # Absolute last resort for Chinese queries
                if is_chinese and len(results) == 0 and 'bot_' in collection_name:
                    # Just return ANY documents for bot collections
                    print("*** LAST RESORT: RETRIEVING SAMPLE DOCUMENTS FROM BOT COLLECTION ***")
                    try:
                        # Get any documents from collection
                        query_sample = vectorstore._collection.peek(limit=10)
                        if query_sample and len(query_sample['documents']) > 0:
                            from langchain_core.documents import Document
                            
                            # Create Document objects from raw docs
                            sample_docs = []
                            for i, doc in enumerate(query_sample['documents']):
                                if i < 5:  # Limit to 5 samples
                                    metadata = {}
                                    # Add metadata if available
                                    if query_sample['metadatas'] and len(query_sample['metadatas']) > i:
                                        metadata = query_sample['metadatas'][i]
                                    sample_docs.append(Document(page_content=doc, metadata=metadata))
                            
                            results = sample_docs
                            print(f"Retrieved {len(results)} sample documents as last resort")
                    except Exception as e:
                        print(f"Error in last resort retrieval: {e}")
                        results = []
                
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
            
        # Skip vectorstore operations if we already have results from direct collection access
        if not results and collections_to_search:
            # If we have no results but it's a Chinese query, use a more aggressive fallback
            if is_chinese:
                update_progress(operation_id, current_step=6.5, 
                               description="No results from initial search, trying aggressive Chinese fallback")
                print(f"No results found, using aggressive Chinese fallback")
                
                # Try direct keyword search on the collection with individual characters
                fallback_collection = collections_to_search[0]
                try:
                    vectorstore = Chroma(
                        persist_directory=CHROMA_PERSIST_DIR, 
                        embedding_function=ollama_ef, 
                        collection_name=fallback_collection
                    )
                    
                    # Extract all Chinese characters from the query
                    chinese_chars = [c for c in query if '\u4e00' <= c <= '\u9fff']
                    print(f"Extracted {len(chinese_chars)} Chinese characters for direct search: {chinese_chars}")
                    
                    # Try searching with each individual character
                    all_fallback_results = []
                    for char in chinese_chars:
                        try:
                            # Get 5 docs per character
                            char_results = vectorstore.similarity_search(char, k=5, score_threshold=0.01)
                            if char_results:
                                print(f"Found {len(char_results)} results for character '{char}'")
                                all_fallback_results.extend(char_results)
                        except Exception as e:
                            print(f"Error in character search for '{char}': {e}")
                    
                    # Deduplicate results
                    unique_docs = {}
                    for doc in all_fallback_results:
                        # Use content as key for deduplication
                        content_hash = doc.page_content[:100]  # First 100 chars as hash
                        if content_hash not in unique_docs:
                            unique_docs[content_hash] = doc
                    
                    # Use these as our results
                    if unique_docs:
                        results = list(unique_docs.values())[:n_results]  # Limit to n_results
                        print(f"Chinese fallback found {len(results)} documents")
                except Exception as e:
                    print(f"Error in Chinese fallback: {e}")
            
            # For non-Chinese queries or if Chinese fallback didn't work
            elif not results and collections_to_search:
                update_progress(operation_id, current_step=6.5, 
                               description="No results from search, but we don't use general fallbacks anymore")
                print(f"No results found in specified collections, and general fallbacks are disabled")
                # Empty results with no fallback
                
        # Update progress
        update_progress(operation_id, current_step=7, 
                       description=f"Retrieved a total of {len(results)} documents")
        
        # We already detected Chinese language at the beginning of the function
        # No need to detect again here
        
        # 2. Remove stopwords and normalize
        import re
        from string import punctuation
        
        # We already processed the query at the beginning - no need to do it again
        # Just use the existing processed_query variable
        
        print(f"Querying vector store for: '{query[:50]}...'")
        print(f"Query language detected: {'Chinese' if is_chinese else 'Other'}")
        print(f"Processed query: '{processed_query[:50]}...'")
        
        update_progress(operation_id, current_step=8, 
                       description=f"Processing query: '{query[:30]}...'" + (" (Chinese text detected)" if is_chinese else ""),
                       details={"is_chinese": is_chinese, "query_sample": query[:50], "processed_query": processed_query[:50]})
        
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
        
        # For multi-collection searches, organize by collection
        results_by_collection = {}
        
        # Group by collection if multiple were searched
        for doc in results:
            collection = doc.metadata.get('collection', 'default')
            if collection not in results_by_collection:
                results_by_collection[collection] = []
            results_by_collection[collection].append(doc)
            
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
                    score += 1.5  # Increased from 1.0 to 1.5 for stronger exact match boost
                    
                    # Additional depth: look for sections with concentrated matches
                    sections = doc.page_content.split('\n\n')
                    for section in sections:
                        if query in section:
                            # If a specific section has the query, boost it further
                            section_ratio = len(section)/len(doc.page_content)
                            score += 0.5 * section_ratio  # More boost for concentrated sections
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
