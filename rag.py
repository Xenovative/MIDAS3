import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, UnstructuredMarkdownLoader, UnstructuredXMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from config import OLLAMA_HOST

# --- Configuration ---
DEFAULT_EMBED_MODEL = "nomic-embed-text"
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "db", "chroma_db")
DEFAULT_COLLECTION_NAME = "rag_documents"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Function to get conversation-specific collection name
def get_conversation_collection_name(conversation_id):
    """Generate a collection name for a specific conversation"""
    return f"conversation_{conversation_id}"

# Ensure ChromaDB persistence directory exists
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# --- LangChain Embedding Function ---
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
    
    loader_xml = DirectoryLoader(source_directory, glob="**/*.xml", loader_cls=UnstructuredXMLLoader, recursive=True, show_progress=True)
    documents.extend(loader_xml.load())

    print(f"Loaded {len(documents)} documents.")
    return documents

def split_documents(documents):
    """Splits documents into smaller chunks."""
    print("Splitting documents...")
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

        # Load the single document based on its extension
        loader = None
        if file_extension == '.txt':
            loader = TextLoader(file_path)
        elif file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension == '.md':
            loader = UnstructuredMarkdownLoader(file_path)
        elif file_extension == '.xml':
            loader = UnstructuredXMLLoader(file_path)
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

def retrieve_context(query, collection_name=DEFAULT_COLLECTION_NAME, conversation_id=None, n_results=3):
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
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": n_results})
        
        print(f"Querying vector store for: '{query[:50]}...'")
        # Use the retriever to get relevant documents
        results = retriever.get_relevant_documents(query)
        
        context_list = [doc.page_content for doc in results]
        context = "\n\n".join(context_list)
        
        print(f"Retrieved {len(context_list)} context chunks.")
        return context
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
