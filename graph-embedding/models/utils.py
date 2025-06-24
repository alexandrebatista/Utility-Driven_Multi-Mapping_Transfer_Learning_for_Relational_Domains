import hashlib
import json
import pickle
from uuid import UUID, uuid5


def set_to_uuid(s):
    namespace = UUID('00000000-0000-0000-0000-000000000000')
    key_base = json.dumps(sorted(s), sort_keys=True).encode('utf-8')
    return str(uuid5(namespace, hashlib.sha1(key_base).hexdigest()))

def save_to_pickle(filepath, obj):
    """
    Saves the given object to a pickle file at the specified filepath.

    Args:
        filepath (str): The path where the pickle file will be saved.
        obj: The Python object to be pickled and saved.
    """
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filepath):
    """
    Loads a Python object from a pickle file.

    Args:
        filepath (str): The path to the pickle file.

    Returns:
        The Python object loaded from the file.
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)