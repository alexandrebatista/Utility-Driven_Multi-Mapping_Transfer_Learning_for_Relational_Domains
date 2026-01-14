import logging
import os
import re
from typing import Optional

from srlearn import Database

from mapping.mapping import parse_modes_file

from .models import DomainMetadata, PrologData

def create_folder(path):
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

def graph_data_to_database(data, domain_metadata: DomainMetadata):
    database = Database()
    database.facts = convert_to_prolog_data(data["facts"])
    database.pos = convert_to_prolog_data(data["pos"])
    database.neg = convert_to_prolog_data(data["neg"])
    database.modes = domain_metadata["raw_modes"]

    return database

def convert_to_prolog_data(data) -> PrologData:
    facts = []
    for pred, tuples in data.items():
        for args in tuples:
            fact = f"{pred}({', '.join(args)})."
            facts.append(fact)
    return facts

def raw_data_to_database(data: PrologData, domain_metadata: DomainMetadata):
    database = Database()
    database.facts = data["facts"]
    database.pos = data["pos"]
    database.neg = data["neg"]
    database.modes = domain_metadata["raw_modes"]

    return database


def extract_domain_metadata(domain_data_path: str) -> DomainMetadata:
    """
    Extracts metadata from a domain dataset located at domain_data_path.
    It parses the modes.pl and pos.pl files to collect predicates, types, the target predicate,
    and also collects the raw mode lines as a single list.

    Returns:
        dict: A dictionary containing modes, predicates, types, target_predicate, and raw_modes.
    """

    modes_path = os.path.join(domain_data_path, "fold00", "modes.pl")
    pos_path = os.path.join(domain_data_path, "fold00", "pos.pl")

    modes = {}
    types = set()
    predicates = set()
    raw_modes = []

    # Parse modes.pl and collect raw lines as a list
    with open(modes_path, "r") as f:
        for line in f:
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith("%"):
                continue

            raw_modes.append(stripped_line)  # add the raw mode line

            # Parse normalized predicate and arguments (remove + and -)
            normalized_line = stripped_line.replace("+", "").replace("-", "")
            match = re.match(r"^(\w+)\(([^)]*)\)\.", normalized_line)
            if not match:
                continue

            predicate = match.group(1)
            args = [arg.strip() for arg in match.group(2).split(",")]

            modes[predicate] = args
            predicates.add(predicate)
            types.update(args)

    # Parse pos.pl to find the target predicate (assumes first occurrence)
    target_predicate = None
    with open(pos_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            match = re.match(r"^(\w+)\(", line)
            if match:
                target_predicate = match.group(1)
                break

    return {
        "modes": modes,
        "predicates": list(predicates),
        "types": list(types),
        "target_predicate": target_predicate,
        "raw_modes": raw_modes
    }

def read_target_domain(domain_path:str, folds: str=None):
    """
    Reads domain data from facts.pl, pos.pl, and neg.pl across folds.

    - facts_data: all predicates except the target predicate from facts.pl
    - pos_data: dict with target predicate as key and set of tuples as value
    - neg_data: dict with target predicate as key and set of tuples as value

    Returns:
        {
            "facts": dict[predicate -> set of tuples],
            "pos": dict[target_predicate -> set of tuples],
            "neg": dict[target_predicate -> set of tuples]
        }
    """
    modes = parse_modes_file(os.path.join(domain_path, "fold00", "modes.pl"))

    folds = folds if folds is not None else sorted(os.listdir(domain_path))

    # Identify target predicate by reading pos.pl of fold00
    target_predicate = None
    pos_file_fold00 = os.path.join(domain_path, "fold00", "pos.pl")
    if os.path.exists(pos_file_fold00):
        with open(pos_file_fold00, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("%") or "(" not in line:
                    continue
                target_predicate = line.split("(")[0]
                break

    # Initialize facts_data excluding the target predicate
    dataset = {
        "facts": {pred: set() for pred in modes if pred != target_predicate},
        "pos": {target_predicate: set()},
        "neg": {target_predicate: set()}
    }
    tmp = []

    for fold in folds:
        fold_path = os.path.join(domain_path, fold)
        if not os.path.isdir(fold_path) or not fold.startswith("fold"):
            continue

        for filename in ["facts.pl", "pos.pl", "neg.pl"]:

            file_path = os.path.join(fold_path, filename)
            if not os.path.exists(file_path):
                continue

            with open(file_path, "r") as infile:
                for line in infile:

                    line = line.strip()
                    if not line or line.startswith("%") or "(" not in line:
                        print(f"The following line couldn't be parsed: {line}")
                        continue

                    predicate = line.split("(")[0]
                    args = line.split("(")[1][:-2].split(",")

                    if predicate not in modes:
                        print(f"{predicate} not found in modes")
                        continue

                    data_type = filename[:-3]
                    if data_type in dataset and predicate in dataset[data_type]:
                        if filename == "pos.pl":
                            tmp.append(tuple(args))
                        dataset[data_type][predicate].add(tuple(args))
                    else:
                        print(f"The following line couldn't be processed: {line}")
    return dataset#, tmp

def load_database(
        path: str = None,
        use_recursion: bool = False,
        target_predicate: str = None,
        reset_target_predicate: bool = False,
        neg_pos_ratio=1,
        max_failed_neg_sampling_retries: Optional[int] = 50,
        logger: logging.Logger = None
):
    if not logger:
        logger = logging.getLogger("Database loader")

    database = Database().fromFiles(
        facts=f"{path}/facts.pl",
        pos=f"{path}/pos.pl",
        neg=f"{path}/neg.pl",
        modes=f"{path}/modes.pl",
        useRecursion=use_recursion
    )
    logger.debug(f"Database loaded with success.")

    if reset_target_predicate:
        logger.warn(
            f"`resetTargetPredicate` is set to True. Target predicate `{target_predicate}` will be ignored and target relation will be set to `None`. Set `resetTargetPredicate` to False if it is desired to set `targetPredicate` as the target relation.")
        database = database.resetTargetPredicate()
    else:
        if target_predicate:
            database = database.setTargetPredicate(
                target_predicate,
                useRecursion=use_recursion,
                negPosRatio=neg_pos_ratio,
                maxFailedNegSamplingRetries=max_failed_neg_sampling_retries
            )
        else:
            default_target_predicate = database.getTargetRelation()
            if default_target_predicate:
                database.neg = database.generateNegativePredicates(
                    negPosRatio=neg_pos_ratio,
                    maxFailedSamplingRetries=max_failed_neg_sampling_retries,
                    keepCurrentNegativeExamples=True
                )
                logger.warn(
                    f"No target predicate was given. It will be set as `{default_target_predicate}` by default. Set `targetPredicate` if it is desired to set another relation as the target relation.")
            else:
                logger.warn(
                    f"No target predicate was given. It will be set as `None` by default. Set `targetPredicate` if it is desired to set any relation as the target relation.")

    return database

def load_test_database(domain_path, folds):
    domain_metadata = extract_domain_metadata(domain_path)

    out = {
        "facts": {predicate: set() for predicate in domain_metadata if predicate != domain_metadata["target_predicate"]},
        "pos": {domain_metadata["target_predicate"]: set()},
        "neg": {domain_metadata["target_predicate"]: set()}
    }

    for name in os.listdir(domain_path):
        if os.path.isdir(domain_path) and name in folds:
            os.path.join(domain_path, name)

            domain_data = read_target_domain(domain_path, folds=["fold00"])

            for data_type in out:
                for predicate in out[data_type]:
                    out[data_type][predicate] = out[data_type][predicate].union(domain_data[data_type][predicate])

    return out

def save_data(path, data):
    create_folder(path)

    for data_type in data:
        with open(os.path.join(path, f"{data_type}.pl"), "w") as f:
            for predicate, args_list in data[data_type].items():
                for args in args_list:
                    f.write(f"{predicate}({','.join(args)}).\n")
