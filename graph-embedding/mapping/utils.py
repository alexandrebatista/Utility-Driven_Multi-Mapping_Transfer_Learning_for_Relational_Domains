import os
import re
from pathlib import Path


def create_folder(path):
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

def save_to_pickle(filepath, obj):
    """
    Saves the given object to a pickle file at the specified filepath.

    Args:
        filepath (str): The path where the pickle file will be saved.
        obj: The Python object to be pickled and saved.
    """
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)