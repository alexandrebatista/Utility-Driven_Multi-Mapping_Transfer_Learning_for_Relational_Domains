import hashlib
import json
import os
import pickle
from uuid import UUID, uuid5
import re


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

def get_hard_samples(path):
    from collections import defaultdict

    samples = defaultdict(set)

    with open(path, 'r') as f:
        for row in f:
            row = row.strip().rstrip('.')  # remove espaços e ponto final
            if '(' in row and ')' in row:
                predicate, arguments = row.split('(', 1)
                arguments = arguments.rstrip(')')
                args = tuple(arg.strip() for arg in arguments.split(','))

                samples[predicate].add(args)

    # Se quiser converter para dict normal:
    return dict(samples)

def load_scores(file_path):
    import pandas as pd

    positive_data = []
    negative_data = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Split into text and score
            parts = line.split()
            score = float(parts[-1])  # last element is the score
            text = " ".join(parts[:-1])  # everything else is the instance

            # Check if it's positive or negative
            if text.startswith("!"):
                negative_data.append({"instance": text[1:], "score": score})
            else:
                positive_data.append({"instance": text, "score": score})

    df_positive = pd.DataFrame(positive_data)
    df_negative = pd.DataFrame(negative_data)

    return df_positive, df_negative

def create_folder(path):
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

def split_and_save_samples_pl(filepath: str, output_dir: str = "."):
    import os
    import pandas as pd

    """
    Splits samples from a .pl file into easy/hard based on score > 0.5,
    separates positives and negatives, removes '!' and scores, and
    saves them into .pl files.
    """
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split last element as score
            *sample_parts, score_str = line.split()
            sample = " ".join(sample_parts)
            score = float(score_str)
            is_positive = not sample.startswith("!")
            if not is_positive:
                sample = sample[1:]  # remove '!' from negative samples
            data.append((sample, score, is_positive))

    # Create DataFrame
    df = pd.DataFrame(data, columns=["sample", "score", "is_positive"])

    # Create directories
    easy_dir = os.path.join(output_dir, "easy")
    hard_dir = os.path.join(output_dir, "hard")
    os.makedirs(easy_dir, exist_ok=True)
    os.makedirs(hard_dir, exist_ok=True)

    # Split easy/hard
    df_easy = df[df["score"] > 0.5]
    df_hard = df[df["score"] < 0.5]

    # Save easy
    df_easy[df_easy["is_positive"]]["sample"].to_csv(os.path.join(easy_dir, "pos.pl"),
                                                     index=False, header=False)
    df_easy[~df_easy["is_positive"]]["sample"].to_csv(os.path.join(easy_dir, "neg.pl"),
                                                      index=False, header=False)

    # Save hard
    df_hard[df_hard["is_positive"]]["sample"].to_csv(os.path.join(hard_dir, "pos.pl"),
                                                     index=False, header=False)
    df_hard[~df_hard["is_positive"]]["sample"].to_csv(os.path.join(hard_dir, "neg.pl"),
                                                      index=False, header=False)

    print(f"Files saved in '{easy_dir}' and '{hard_dir}'")


def generate_folds(base_path: str, small_dataset=True, use_validation=False):
    """
    Generate train/test fold splits from directories following the 'foldXX' pattern.

    Args:
        use_validation: Select one validation fold randomly.
        base_path (str): Path containing the fold directories.
        small_dataset (bool): Set if the training dataset has 1 fold or n - 1 folds

    Returns:
        list[dict]: A list of dictionaries, each containing 'train' and 'test' keys.
    """
    import random

    # Get all directories that start with "fold"
    folds = sorted([d for d in os.listdir(base_path) if d.startswith("fold")])

    # Create all train/test splits
    if small_dataset:
        splits = [
            {"train": [fold], "test": [f for f in folds if f != fold]}
            for fold in folds
        ]
        if use_validation:
            for configuration in splits:
                validation_fold = random.choice(configuration["test"])
                configuration["test"].remove(validation_fold)
                configuration["validation"] = [validation_fold]
    else:
        splits = [
            {"test": [fold], "train": [f for f in folds if f != fold]}
            for fold in folds
        ]
        if use_validation:
            for configuration in splits:
                validation_fold = random.choice(configuration["train"])
                configuration["train"].remove(validation_fold)
                configuration["validation"] = [validation_fold]

    return splits

def update_experiment_state(file_path: str, new_results: dict):
    """
    Reads the current experiment state from a JSON file, updates it with new results,
    and writes it back to disk.

    Args:
        file_path (str): Path to the JSON file.
        new_results (dict): Dictionary containing new experiment results.
    """
    # Load existing data (if the file exists)
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Update the state with new results
    data.append(new_results)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save updated data back to JSON
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)