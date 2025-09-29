import hashlib
import json

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling NumPy arrays and other NumPy types"""

    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)

        # Handle Environment and Belief objects by using their config_id
        if hasattr(o, "config_id"):
            try:
                return {
                    "__class__": o.__class__.__name__,
                    "__module__": o.__class__.__module__,
                    "__config_id__": o.config_id,
                }
            except Exception:
                # If config_id fails, fall through to other methods
                pass

        # Handle ActionSampler instances by converting to their state
        if hasattr(o, "__class__") and hasattr(o, "__getstate__"):
            try:
                # Try to get the serializable state
                state = o.__getstate__()
                # Add class information for reconstruction
                return {
                    "__class__": o.__class__.__name__,
                    "__module__": o.__class__.__module__,
                    "__state__": state,
                }
            except Exception:
                # If serialization fails, use string representation
                return str(o)

        return super().default(o)


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
    hash_o = hashlib.sha256(dict_str.encode())

    # Return the hexadecimal representation of the hash
    return hash_o.hexdigest()
