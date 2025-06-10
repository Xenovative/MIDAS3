from typing import List, Dict, Tuple, Optional
import logging
import os
import json
from serpapi import GoogleSearch
import requests.exceptions
from urllib.parse import urlparse

# Load configuration
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'user_preferences.json')

def get_serpapi_key() -> Optional[str]:
    """Get SerpAPI key from config file or environment variable"""
    # Check environment variable first
    if 'SERPAPI_KEY' in os.environ:
        return os.environ['SERPAPI_KEY']
    
    # Check config file
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                return config.get('serpapi_key')
    except Exception as e:
        logging.error(f"Error reading config file: {e}")
    
    return None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_get(dictionary, *keys, default=''):
    """Safely get nested dictionary keys"""
    for key in keys:
        try:
            dictionary = dictionary[key]
        except (KeyError, TypeError, AttributeError):
            return default
    return dictionary

def search_web(query: str, max_results: int = 5, max_retries: int = 3) -> Tuple[List[Dict[str, str]], bool]:
    """
    Perform a web search using SerpAPI with retries and return results.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return (1-10)
        max_retries: Maximum number of retry attempts
        
    Returns:
        Tuple of (results, success) where success indicates if the search was successful
    """
    if not query or not query.strip():
        logger.warning("Empty search query provided")
        return [], False
    
    # Get API key
    api_key = get_serpapi_key()
    if not api_key:
        logger.error("SerpAPI key not found. Please set SERPAPI_KEY environment variable or add 'serpapi_key' to config.json")
        return [], False
    
    query = query.strip()
    logger.info(f"Performing web search for: {query}")
    
    for attempt in range(max_retries):
        try:
            # Configure the search
            params = {
                'q': query,
                'api_key': api_key,
                'num': min(max_results, 10),  # SerpAPI max is 100 but we'll limit to 10
                'hl': 'en',
                'gl': 'us',
                'safe': 'active'
            }
            
            search = GoogleSearch(params)
            results_data = search.get_dict()
            
            # Parse the results
            results = []
            
            # Handle organic results
            if 'organic_results' in results_data:
                for result in results_data['organic_results']:
                    if len(results) >= max_results:
                        break
                        
                    title = result.get('title', '')
                    link = result.get('link', '')
                    snippet = result.get('snippet', '')
                    
                    # Skip if we don't have required fields
                    if not all([title, link]):
                        continue
                    
                    # Get favicon if available
                    favicon = result.get('favicon', '')
                    
                    results.append({
                        'title': title,
                        'link': link,
                        'snippet': snippet,
                        'source': urlparse(link).netloc,
                        'favicon': favicon
                    })
            
            if results:
                logger.info(f"Found {len(results)} results")
                return results, True
                
            logger.warning(f"No valid results found in attempt {attempt + 1}")
            
        except Exception as e:
            error_type = type(e).__name__
            logger.error(f"{error_type} during web search (attempt {attempt + 1}): {str(e)}", exc_info=True)
            if attempt == max_retries - 1:  # Last attempt
                return [], False
            
            # Exponential backoff
            time.sleep(2 ** attempt)
    
    logger.error(f"Failed to get web results after {max_retries} attempts")
    return [], False

def format_web_results(results: List[Dict[str, str]]) -> str:
    """Format web search results into a well-structured string with sources"""
    if not results:
        return "No relevant web results found. Please try rephrasing your query or check your internet connection."
    
    try:
        formatted = ["🌐 **Web Search Results**\n"]
        
        for i, result in enumerate(results, 1):
            # Get result data with fallbacks
            title = result.get('title', 'No title').strip()
            link = result.get('link', '').strip()
            snippet = result.get('snippet', 'No description available.').strip()
            source = result.get('source', '').strip()
            favicon = result.get('favicon', '')
            
            # Truncate long snippets
            if len(snippet) > 200:
                snippet = snippet[:197] + '...'
            
            # Format the result
            formatted_result = [
                f"{i}. **{title}**"
            ]
            
            # Add favicon if available
            if favicon:
                formatted_result[0] = f"{formatted_result[0]} <img src='{favicon}' height='12'/>"
            
            # Add source and link
            source_line = []
            if source:
                source_line.append(f"🔗 {source}")
            if link:
                source_line.append(f"[View Source]({link})")
                
            formatted_result.append(f"   📌 {snippet}")
            if source_line:
                formatted_result.append(f"   {' | '.join(source_line)}")
                
            formatted_result.append("")
            formatted.extend(formatted_result)
            
        return '\n'.join(formatted)
        
    except Exception as e:
        logger.error(f"Error formatting web results: {str(e)}", exc_info=True)
        return "Error formatting web search results. Please try again later."

def get_web_context(query: str, max_results: int = 5) -> str:
    """
    Get web context for a query with enhanced error handling
    
    Args:
        query: The search query
        max_results: Maximum number of results to return (1-10)
        
    Returns:
        Formatted string with web search results or error message
    """
    try:
        # Validate max_results
        max_results = max(1, min(10, int(max_results)))  # Ensure between 1-10
        
        logger.info(f"Getting web context for query: {query}")
        
        # Clean the query
        clean_query = ' '.join(str(query).strip().split())
        if not clean_query:
            return "Error: Empty search query"
            
        # Perform the search
        results, success = search_web(clean_query, max_results)
        
        if not success or not results:
            return ("I couldn't retrieve any web results for that query. "
                   "This might be due to network issues or the search service being temporarily unavailable. "
                   "Please try again in a moment.")
        
        # Format and return results
        return format_web_results(results)
        
    except Exception as e:
        logger.error(f"Error in get_web_context: {str(e)}", exc_info=True)
        return ("I encountered an error while searching the web. "
               "This might be a temporary issue. Please try again later.")
