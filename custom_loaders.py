"""
Custom document loaders for MIDAS3
"""
import os
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

class ChineseTheologyXMLLoader:
    """
    Custom loader for Chinese Theology XML files with specific tag structure:
    <title>, <author>, <periodical>, <content>
    
    This loader extracts structured data from XML and creates better document chunks
    with rich metadata for improved retrieval.
    """
    
    def __init__(self, file_path: str):
        """Initialize with path to XML file."""
        self.file_path = file_path
    
    def load(self) -> List[Document]:
        """
        Load and parse the XML file, returning a list of Documents.
        Each document contains content with appropriate metadata from XML tags.
        """
        try:
            # Parse the XML file
            tree = ET.parse(self.file_path)
            root = tree.getroot()
            
            documents = []
            
            # Extract common metadata that applies to all chunks
            title = self._get_tag_text(root, 'title', 'Untitled')
            author = self._get_tag_text(root, 'author', 'Unknown Author')
            periodical = self._get_tag_text(root, 'periodical', 'Unknown Publication')
            
            # Process the content tag which contains the main text
            content_elem = root.find('content')
            
            if content_elem is not None and content_elem.text:
                # Create the main document with full content
                main_doc = Document(
                    page_content=content_elem.text,
                    metadata={
                        'source': self.file_path,
                        'title': title,
                        'author': author,
                        'periodical': periodical,
                        'type': 'full_content'
                    }
                )
                documents.append(main_doc)
                
                # Also create a title document with rich metadata for better retrieval
                # This helps when users search for articles by title
                title_doc = Document(
                    page_content=f"Title: {title}\nAuthor: {author}\nPublication: {periodical}",
                    metadata={
                        'source': self.file_path,
                        'title': title,
                        'author': author,
                        'periodical': periodical,
                        'type': 'metadata'
                    }
                )
                documents.append(title_doc)
            
            # If there are section tags within content, create separate documents for each
            sections = root.findall('.//section')
            for i, section in enumerate(sections):
                if section.text and section.text.strip():
                    # Get section title if available
                    section_title = section.get('title', f'Section {i+1}')
                    
                    section_doc = Document(
                        page_content=section.text,
                        metadata={
                            'source': self.file_path,
                            'title': title,
                            'section': section_title,
                            'author': author,
                            'periodical': periodical,
                            'type': 'section'
                        }
                    )
                    documents.append(section_doc)
            
            # If no sections were found but we have paragraphs, use those
            if not sections:
                paragraphs = root.findall('.//p')
                for i, para in enumerate(paragraphs):
                    if para.text and para.text.strip():
                        para_doc = Document(
                            page_content=para.text,
                            metadata={
                                'source': self.file_path,
                                'title': title,
                                'paragraph': i+1,
                                'author': author,
                                'periodical': periodical,
                                'type': 'paragraph'
                            }
                        )
                        documents.append(para_doc)
            
            # If we didn't get any documents from sections or paragraphs,
            # and the main content is too long, split it into chunks
            if len(documents) <= 2 and content_elem is not None and content_elem.text:
                content_text = content_elem.text
                # Simple chunking by newlines if the content is long
                if len(content_text) > 2000:
                    chunks = content_text.split('\n\n')
                    for i, chunk in enumerate(chunks):
                        if chunk.strip():
                            chunk_doc = Document(
                                page_content=chunk,
                                metadata={
                                    'source': self.file_path,
                                    'title': title,
                                    'chunk': i+1,
                                    'total_chunks': len(chunks),
                                    'author': author,
                                    'periodical': periodical,
                                    'type': 'chunk'
                                }
                            )
                            documents.append(chunk_doc)
            
            print(f"Extracted {len(documents)} documents from {os.path.basename(self.file_path)}")
            return documents
            
        except Exception as e:
            print(f"Error parsing XML file {self.file_path}: {e}")
            # Return an empty document with error information
            return [Document(
                page_content=f"Error parsing XML file: {e}",
                metadata={'source': self.file_path, 'error': str(e)}
            )]
    
    def _get_tag_text(self, root, tag_name, default=""):
        """Helper to safely extract text from an XML tag."""
        element = root.find(tag_name)
        if element is not None and element.text:
            return element.text.strip()
        return default


class ChineseTheologyXMLDirectoryLoader:
    """
    Loads all XML files in a directory using the ChineseTheologyXMLLoader.
    """
    
    def __init__(self, directory_path: str, glob_pattern: str = "**/*.xml", recursive: bool = True):
        """Initialize with directory path and optional glob pattern."""
        self.directory_path = directory_path
        self.glob_pattern = glob_pattern
        self.recursive = recursive
    
    def load(self) -> List[Document]:
        """
        Load all XML files in the directory that match the glob pattern.
        """
        import glob
        
        documents = []
        
        # Generate file paths based on glob pattern
        if self.recursive:
            matches = glob.glob(os.path.join(self.directory_path, self.glob_pattern), recursive=True)
        else:
            matches = glob.glob(os.path.join(self.directory_path, self.glob_pattern))
        
        # Load each file
        for file_path in matches:
            if os.path.isfile(file_path) and file_path.lower().endswith('.xml'):
                try:
                    loader = ChineseTheologyXMLLoader(file_path)
                    file_docs = loader.load()
                    documents.extend(file_docs)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    # Add an error document
                    documents.append(Document(
                        page_content=f"Error loading XML file: {e}",
                        metadata={'source': file_path, 'error': str(e)}
                    ))
        
        print(f"Loaded {len(documents)} total documents from {len(matches)} XML files")
        return documents
