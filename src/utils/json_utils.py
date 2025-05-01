"""
JSON Utilities for the Real-Time Task Scheduling Project

This module provides utilities for JSON serialization, including a custom
JSON encoder that can handle NumPy data types.
"""

import json
import numpy as np
from typing import Any, Dict, List, Optional, Union

class NumpyJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles NumPy types.
    
    This encoder converts NumPy data types to their standard Python equivalents
    so they can be properly serialized to JSON. It handles:
    - numpy integers (np.int8, np.int16, np.int32, np.int64, etc.)
    - numpy floats (np.float16, np.float32, np.float64, etc.)
    - numpy booleans (np.bool_, np.bool8)
    - numpy arrays (np.ndarray)
    
    Example usage:
        import json
        from src.utils.json_utils import NumpyJSONEncoder
        
        data = {'array': np.array([1, 2, 3]), 'bool': np.bool_(True)}
        json_str = json.dumps(data, cls=NumpyJSONEncoder)
    """
    def default(self, obj: Any) -> Any:
        """
        Implement custom handling for NumPy types
        
        Args:
            obj: Object to serialize
            
        Returns:
            JSON-serializable equivalent of the object
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, np.bool8)):
            return bool(obj)
        return super().default(obj)

def save_json(data: Any, filepath: str, pretty: bool = True) -> None:
    """
    Save data to a JSON file with NumPy type handling
    
    Args:
        data: Data to save
        filepath: Path to the output file
        pretty: Whether to format with indentation (default: True)
    """
    indent = 4 if pretty else None
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, cls=NumpyJSONEncoder)

def load_json(filepath: str) -> Any:
    """
    Load data from a JSON file
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)