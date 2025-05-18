import hashlib
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling NumPy arrays and other NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

def config_to_id(config_dict: dict) -> str:
    """
    Generate a unique ID from a configuration dictionary using hashing.
    Handles NumPy arrays and other NumPy types.
    
    Args:
        config_dict (dict): The configuration dictionary to hash
        
    Returns:
        str: A unique hash string representing the configuration
    """
    # Sort the dictionary to ensure consistent hashing regardless of key order
    sorted_dict = dict(sorted(config_dict.items()))
    
    # Convert dictionary to a JSON string, handling NumPy types
    dict_str = json.dumps(sorted_dict, sort_keys=True, cls=NumpyEncoder)
    
    # Create a hash of the string using SHA-256
    hash_obj = hashlib.sha256(dict_str.encode())
    
    # Return the hexadecimal representation of the hash
    return hash_obj.hexdigest()
