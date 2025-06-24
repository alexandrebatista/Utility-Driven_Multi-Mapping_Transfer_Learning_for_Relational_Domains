import json
import os
import pickle
from pathlib import Path

from srlearn.rdn import RDNBoost

from database.database import DatabaseWrapper
from database.utils import extract_domain_metadata, read_target_domain
from mapping.utils import create_folder
from models.selector_model import SelectorModel
from models.utils import set_to_uuid, save_to_pickle, load_pickle


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


def _model_exists(predicates):
    pass


def _get_models_metadata(metadata_path):
    if not os.path.exists(metadata_path):
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({}, f)

        return {}

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    return metadata

def _search_model_metadata(metadata, predicates):
    model_uuid = set_to_uuid(predicates)

    return metadata.get(model_uuid)

def _update_model_metadata(metadata_path, metadata):
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)

def selector_models_workflow(
        data_path,
        mappings_path,
        selector_models_path):

    models_path = os.path.join(selector_models_path, "models")
    models_metadata_path_dir = os.path.join(selector_models_path, "metadata")

    create_folder(models_path)
    create_folder(models_metadata_path_dir)

    for mappings_file in os.listdir(mappings_path):
        if mappings_file.endswith('.json'):
            file_path = os.path.join(mappings_path, mappings_file)
            print(f'  Reading file: {mappings_file}')

            # Open and load the JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                mappings = json.load(f)

            for mapping_id, mapping in mappings.items():

                predicates = set(
                    [predicate for predicate in mapping["predicate_mappings"].values() if predicate is not None])

                model_metadata_path_file = os.path.join(models_metadata_path_dir, f"{mapping["target"]}.json")

                models_metadata = _get_models_metadata(model_metadata_path_file)
                model_metadata = _search_model_metadata(models_metadata, predicates)

                if model_metadata is None:
                    target_domain_path = os.path.join(data_path, mapping["target"])

                    create_folder(os.path.join(models_path, mapping["target"]))

                    target_domain_metadata = extract_domain_metadata(target_domain_path)

                    selector_model = SelectorModel(valid_predicates=predicates,
                                                   source=mapping["source"],
                                                   target=mapping["target"],
                                                   target_domain_metadata=target_domain_metadata)


                    target_domain_data = read_target_domain(target_domain_path)
                    selector_model.train(target_domain_data)

                    # Create model
                    models_metadata[set_to_uuid(predicates)] = selector_model.get_metadata()
                    _update_model_metadata(model_metadata_path_file, models_metadata)
                    save_to_pickle(os.path.join(models_path, mapping["target"], set_to_uuid(predicates)), selector_model)
                else:
                    # load model
                    selector_model = load_pickle(os.path.join(models_path, mapping["target"], set_to_uuid(predicates)))

                break
                test_metrics = selector_model.test(target_domain_data)

