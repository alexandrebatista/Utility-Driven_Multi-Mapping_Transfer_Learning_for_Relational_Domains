import json
import os
import pickle
from pathlib import Path

from srlearn.rdn import RDNBoost

from database.database import DatabaseWrapper
from mapping.utils import create_folder


def reference_models_workflow(train_folds="fold00"):
    base_dir = "/home/alexandre/mestrado/graph-embedding/preprocessed"
    pickles_models_path = f'/home/alexandre/mestrado/graph-embedding/pickle_models/'

    path = Path(base_dir)

    for target_domain_path in path.iterdir():
        if train_folds is None:
            train_folds = ["fold00"]

        if os.path.exists(path):
            print("Model pickle file exists.")
            continue

        model_path = os.path.join(pickles_models_path, f"reference_{target_domain_path.name}_{train_folds}.pkl")

        database = DatabaseWrapper()

        database.process_target(target_domain_path)

        train_database = database.generate_train_srldatabase(train_folds, use_recursion=True)

        settings = {
            "randomSeed": 11,
            "numberOfClauses": 8,
            "numberOfCycles": 100,
            "maxTreeDepth": 3,
            "nEstimators": 5,
            "nodeSize": 2,
            "neg_pos_ratio": 2,
            "max_failed_neg_sampling_retries": 50,
            "ignoreSTDOUT": True,
            "useRecursion": True,
            "runOriginalRDNBoost": False
        }

        ignore_stdout = True

        model = RDNBoost(
            n_estimators=settings["nEstimators"],
            node_size=settings["nodeSize"],
            max_tree_depth=settings["maxTreeDepth"],
            neg_pos_ratio=settings["neg_pos_ratio"],
            number_of_clauses=settings["numberOfClauses"],
            number_of_cycles=settings["numberOfCycles"],
            path="./"
        )

        model.fit(train_database, ignoreSTDOUT=ignore_stdout)

        model.predict(train_database, threshold=0.5)

        with open(model_path, 'wb') as file:
            pickle.dump(model, file)

def selector_models_workflow(mappings_path, selector_models_path):
    create_folder(os.path.join(selector_models_path, "models"))
    create_folder(os.path.join(selector_models_path, "metadata"))

    for mappings_file in os.listdir(mappings_path):
        if mappings_file.endswith('.json'):
            file_path = os.path.join(mappings_path, mappings_file)
            print(f'  Reading file: {mappings_file}')

            # Open and load the JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                mappings = json.load(f)

            for mapping_id, mapping in mappings.items():
                pass

