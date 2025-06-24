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

import re

def extract_metrics_from_log(log_path):
    """
    Extracts evaluation metrics from the end of a log file.

    Args:
        log_path (str): Path to the .txt log file.

    Returns:
        dict: Dictionary containing extracted metrics.
    """
    metrics = {}

    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Only keep lines that match known metric patterns
    for line in reversed(lines):
        line = line.strip()
        if not line.startswith("%"):
            continue

        if "AUC ROC" in line:
            match = re.search(r"AUC ROC\s*=\s*([0-9,\.]+)", line)
            if match:
                metrics["auc_roc"] = float(match.group(1).replace(",", "."))

        elif "AUC PR" in line:
            match = re.search(r"AUC PR\s*=\s*([0-9,\.]+)", line)
            if match:
                metrics["auc_pr"] = float(match.group(1).replace(",", "."))

        elif "CLL" in line:
            match = re.search(r"CLL\s*=\s*([-\d,\.]+)", line)
            if match:
                metrics["cll"] = float(match.group(1).replace(",", "."))

        elif "Precision" in line:
            match = re.search(r"Precision\s*=\s*([0-9,\.]+)", line)
            if match:
                metrics["precision"] = float(match.group(1).replace(",", "."))

        elif "Recall" in line:
            match = re.search(r"Recall\s*=\s*([0-9,\.]+)", line)
            if match:
                metrics["recall"] = float(match.group(1).replace(",", "."))

        elif "F1" in line:
            match = re.search(r"F1\s*=\s*([0-9,\.]+)", line)
            if match:
                metrics["f1"] = float(match.group(1).replace(",", "."))

        elif "inference time" in line:
            match = re.search(r"([0-9]+)\s+milliseconds", line)
            if match:
                metrics["inference_time_ms"] = int(match.group(1))

        # Stop if all metrics found
        if len(metrics) == 7:
            break

    return metrics

def extract_learning_time_from_log(log_path):
    """
    Extracts the total learning time from a log file.

    Args:
        log_path (str): Path to the log file.

    Returns:
        float or None: Learning time in seconds, or None if not found.
    """
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in reversed(lines):
        line = line.strip()
        if "Total learning time" in line:
            match = re.search(r"Total learning time.*?:\s*([0-9,\.]+)\s*seconds", line)
            if match:
                return float(match.group(1).replace(",", "."))

    return None