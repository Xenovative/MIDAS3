"""
Fast document loaders for improved indexing performance
"""
import os
import logging
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

class FastXMLLoader:
    """A high-performance XML loader that processes XML files much faster than UnstructuredXMLLoader"""
    
    def __init__(self, file_path: str, content_tags: List[str] = None, encoding: str = "utf-8"):
        """
        Initialize the FastXMLLoader.
        
        Args:
            file_path: Path to the XML file
            content_tags: List of XML tags to extract content from (if None, extracts from all text nodes)
            encoding: File encoding
        """
        self.file_path = file_path
        self.content_tags = content_tags or ["p", "text", "content", "body", "section", "div", "entry", "item"]
        self.encoding = encoding
        
    def load(self) -> List[Document]:
        """
        Load and parse the XML file into Documents.
        
        Returns:
            List of Document objects with page content and metadata
        """
        try:
            # Parse the XML file
            tree = ET.parse(self.file_path)
            root = tree.getroot()
            
            # Extract documents
            documents = []
            
            # Process the XML tree
            self._process_element(root, documents, path="")
            
            # If no documents were created, create one with the whole XML content
            if not documents:
                with open(self.file_path, "r", encoding=self.encoding) as f:
                    content = f.read()
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": self.file_path,
                            "filename": os.path.basename(self.file_path),
                        }
                    )
                )
            
            return documents
            
        except Exception as e:
            logging.error(f"Error loading XML file {self.file_path}: {str(e)}")
            # Fallback to returning a single document with the error message
            return [
                Document(
                    page_content=f"Error loading XML file: {str(e)}",
                    metadata={
                        "source": self.file_path,
                        "filename": os.path.basename(self.file_path),
                        "error": str(e)
                    }
                )
            ]
    
    def _process_element(self, element: ET.Element, documents: List[Document], path: str) -> None:
        """
        Recursively process XML elements and extract content.
        
        Args:
            element: Current XML element
            documents: List to append documents to
            path: Current element path for metadata
        """
        # Update path
        current_path = f"{path}/{element.tag}" if path else element.tag
        
        # Check if this is a content tag
        is_content_tag = element.tag in self.content_tags
        
        # Extract text content from this element (excluding child element text)
        element_text = (element.text or "").strip()
        
        # If this is a content tag with text, create a document
        if is_content_tag and element_text:
            # Get attributes as metadata
            metadata = {
                "source": self.file_path,
                "filename": os.path.basename(self.file_path),
                "path": current_path
            }
            
            # Add element attributes to metadata
            for attr_name, attr_value in element.attrib.items():
                metadata[f"attr_{attr_name}"] = attr_value
            
            # Create document
            documents.append(
                Document(
                    page_content=element_text,
                    metadata=metadata
                )
            )
        
        # Process child elements
        for child in element:
            self._process_element(child, documents, current_path)
            
        # Check for tail text (text after the element)
        if element.tail and element.tail.strip():
            # Add tail text to parent element if significant
            parent_path = path
            documents.append(
                Document(
                    page_content=element.tail.strip(),
                    metadata={
                        "source": self.file_path,
                        "filename": os.path.basename(self.file_path),
                        "path": parent_path
                    }
                )
            )
