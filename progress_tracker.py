"""
Progress tracking module for long-running operations in MIDAS3
"""
import threading
import time
import json
from typing import Dict, Any, Optional

# Global dictionary to store progress information for different operations
# Key: operation_id, Value: progress dictionary
_progress_store: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()

def create_progress(operation_id: str, total_steps: int = 100, description: str = "Operation in progress") -> None:
    """
    Create a new progress tracker for an operation
    
    Args:
        operation_id: Unique identifier for the operation
        total_steps: Total number of steps in the operation
        description: Description of the operation
    """
    with _lock:
        _progress_store[operation_id] = {
            "id": operation_id,
            "current_step": 0,
            "total_steps": total_steps,
            "percentage": 0,
            "description": description,
            "status": "in_progress",
            "start_time": time.time(),
            "update_time": time.time(),
            "end_time": None,
            "error": None,
            "details": {}
        }

def update_progress(operation_id: str, current_step: Optional[int] = None, 
                   increment: int = 0, description: Optional[str] = None,
                   status: Optional[str] = None, details: Optional[Dict[str, Any]] = None,
                   error: Optional[str] = None) -> Dict[str, Any]:
    """
    Update the progress of an operation
    
    Args:
        operation_id: Unique identifier for the operation
        current_step: Current step number (if provided, overrides increment)
        increment: Number of steps to increment (ignored if current_step is provided)
        description: New description (if provided)
        status: New status (if provided)
        details: Additional details to update (if provided)
        error: Error message (if provided)
        
    Returns:
        Updated progress dictionary
    """
    with _lock:
        if operation_id not in _progress_store:
            # Create a new progress tracker if it doesn't exist
            create_progress(operation_id)
        
        progress = _progress_store[operation_id]
        
        # Update step count
        if current_step is not None:
            progress["current_step"] = current_step
        else:
            progress["current_step"] += increment
        
        # Ensure current_step doesn't exceed total_steps
        progress["current_step"] = min(progress["current_step"], progress["total_steps"])
        
        # Update percentage
        if progress["total_steps"] > 0:
            progress["percentage"] = int((progress["current_step"] / progress["total_steps"]) * 100)
        
        # Update description if provided
        if description is not None:
            progress["description"] = description
        
        # Update status if provided
        if status is not None:
            progress["status"] = status
            if status == "completed":
                progress["end_time"] = time.time()
                progress["current_step"] = progress["total_steps"]
                progress["percentage"] = 100
            elif status == "error":
                progress["end_time"] = time.time()
        
        # Update error if provided
        if error is not None:
            progress["error"] = error
            progress["status"] = "error"
            progress["end_time"] = time.time()
        
        # Update details if provided
        if details is not None:
            progress["details"].update(details)
        
        # Update timestamp
        progress["update_time"] = time.time()
        
        return progress.copy()

def get_progress(operation_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the current progress of an operation
    
    Args:
        operation_id: Unique identifier for the operation
        
    Returns:
        Progress dictionary or None if not found
    """
    with _lock:
        return _progress_store.get(operation_id, {}).copy()

def get_all_progress() -> Dict[str, Dict[str, Any]]:
    """
    Get all progress trackers
    
    Returns:
        Dictionary of all progress trackers
    """
    with _lock:
        return _progress_store.copy()

def clear_progress(operation_id: str) -> None:
    """
    Clear a progress tracker
    
    Args:
        operation_id: Unique identifier for the operation
    """
    with _lock:
        if operation_id in _progress_store:
            del _progress_store[operation_id]

def clear_completed_progress(older_than_seconds: int = 3600) -> None:
    """
    Clear completed progress trackers older than the specified time
    
    Args:
        older_than_seconds: Clear completed progress older than this many seconds
    """
    current_time = time.time()
    with _lock:
        to_delete = []
        for op_id, progress in _progress_store.items():
            if progress["status"] in ["completed", "error"] and progress["end_time"] is not None:
                if current_time - progress["end_time"] > older_than_seconds:
                    to_delete.append(op_id)
        
        for op_id in to_delete:
            del _progress_store[op_id]

def progress_to_json(operation_id: str) -> str:
    """
    Convert progress to JSON string
    
    Args:
        operation_id: Unique identifier for the operation
        
    Returns:
        JSON string representation of the progress
    """
    progress = get_progress(operation_id)
    if progress:
        return json.dumps(progress)
    return "{}"
