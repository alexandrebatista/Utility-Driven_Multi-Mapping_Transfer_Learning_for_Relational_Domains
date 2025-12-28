import json
import os
import shutil

from srlearn.rdn import RDNBoost

from database.database import DatabaseWrapper

from models.utils import extract_metrics_from_log, create_folder, create_folder, set_to_uuid, load_pickle, save_to_pickle
from .selector_model import SelectorModel
from .utils import create_folder, set_to_uuid, load_pickle, save_to_pickle, generate_folds, update_experiment_state
from models.model import search_best_model
from database.utils import extract_domain_metadata, read_target_domain
from mapping.apply_mappings import read_source_domain_and_apply_mapping

def models_workflow(target_domain: str, mappings_data_path, train_folds: list = None):
    target_domain_path = f"./preprocessed/{target_domain}"
    mappings_path = f"./mappings/{target_domain}.json"
    #mappings_data_path = f"./selected_samples/{target_domain}"

    with open(mappings_path, "r", encoding="utf-8") as f:
        mappings = json.load(f)

    if train_folds is None:
        train_folds = ["fold00"]

    test_folds = []
    for name in os.listdir(target_domain_path):
        if os.path.isdir(os.path.join(target_domain_path, name)) and name not in train_folds:
            test_folds.append(name)

    test_folds.sort()

    # ------------------------------------------------------------------------------------------------------------------
    # Transforming the datasets in Srlearn databases

    database = DatabaseWrapper()
    database.process_mappings(mappings, mappings_data_path)
    database.process_target(target_domain_path)

    train_database = database.generate_tl_database(train_folds, use_recursion=True)
    test_database = database.generate_test_srldatabase(test_folds, use_recursion=False)

    # ------------------------------------------------------------------------------------------------------------------
    # Training the models

    settings = {
        "randomSeed": 11,
        "numberOfClauses": 8,
        "numberOfCycles": 100,
        "maxTreeDepth": 3,
        "nEstimators": 10,
        "nodeSize": 2,
        "neg_pos_ratio": 2,
        "max_failed_neg_sampling_retries": 50,
        "ignoreSTDOUT": True,
        "useRecursion": True,
        "runOriginalRDNBoost": False
    }

    utility_alpha = 1
    model = RDNBoost(
        n_estimators=settings["nEstimators"],
        node_size=settings["nodeSize"],
        max_tree_depth=settings["maxTreeDepth"],
        neg_pos_ratio=settings["neg_pos_ratio"],
        number_of_clauses=settings["numberOfClauses"],
        number_of_cycles=settings["numberOfCycles"],
        source_utility_alpha=utility_alpha,
        target_utility_alpha=utility_alpha,
        path="./"
    )

    ignore_stdout = True

    model.fit(train_database, ignoreSTDOUT=ignore_stdout)

    model.predict(test_database, threshold=0.5)

def instance_selection_workflow(target_domain: str, base_path, mappings):
    def save_mapped_dataset(facts, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for predicate in facts:
                for args in facts[predicate]:
                    f.write(f"{predicate}({','.join(args)}).\n")

    def save_selected_samples(base_path, scores_by_mapping):
        pos = []
        neg = []

        for mapping_id, samples in scores_by_mapping.items():
            for sample in samples:
                if sample["raw"][0] != "!":
                    pos.append({"mapping_id": mapping_id, **sample})
                else:
                    neg.append({"mapping_id": mapping_id, **sample})

        pos = sorted(pos, key=lambda x: x["score"], reverse=True)
        neg = sorted(neg, key=lambda x: x["score"], reverse=True)

        best_pos = [item for item in pos if item["score"] > 0.5]
        worst_pos = [item for item in pos if item["score"] <= 0.5]
        best_neg = [item for item in neg if item["score"] > 0.5]
        worst_neg = [item for item in neg if item["score"] <= 0.5]

        for sample in best_pos:
            with open(f"{base_path}/selected_samples_best/cora/{sample['mapping_id']}/pos.pl", "a",
                      encoding="utf-8") as f:
                f.write(str(sample["raw"] + ".") + "\n")

        for sample in best_neg:
            with open(f"{base_path}/selected_samples_best/cora/{sample['mapping_id']}/neg.pl", "a",
                      encoding="utf-8") as f:
                f.write(str(sample["raw"][1:] + ".") + "\n")

        for sample in best_pos:
            with open(f"{base_path}/selected_samples_worst/cora/{sample['mapping_id']}/pos.pl", "a",
                      encoding="utf-8") as f:
                f.write(str(sample["raw"] + ".") + "\n")

        for sample in best_neg:
            with open(f"{base_path}/selected_samples_worst/cora/{sample['mapping_id']}/neg.pl", "a",
                      encoding="utf-8") as f:
                f.write(str(sample["raw"][1:] + ".") + "\n")

    data_path = "./preprocessed"
    target_domain_path = f"./preprocessed/{target_domain}"
    target_domain_data = read_target_domain(target_domain_path)
    target_domain_metadata = extract_domain_metadata(target_domain_path)

    best_model = search_best_model(target_domain_data, target_domain_metadata, base_path)
    best_model.process_data()

    hardest_samples_path = os.path.join(base_path, "below")

    models_path = os.path.join(base_path, "models")
    models_metadata_path_dir = os.path.join(base_path, "metadata")

    create_folder(models_path)
    create_folder(models_metadata_path_dir)

    scores_by_mapping = {}

    for mapping in mappings:
        mapping_id = mapping["id"]

        predicates = set(
            [predicate for predicate in mapping["predicate_mappings"].values() if predicate is not None])

        model_metadata_path_file = os.path.join(models_metadata_path_dir, f"{mapping['target']}.json")

        models_metadata = SelectorModel.get_models_metadata(model_metadata_path_file)
        model_metadata = SelectorModel.search_model_metadata(models_metadata, predicates)

        target_domain_path = os.path.join(hardest_samples_path, mapping["target"])

        create_folder(os.path.join(models_path, mapping["target"]))
        target_domain_metadata = extract_domain_metadata(hardest_samples_path)

        if model_metadata is None:
            selector_model = SelectorModel(valid_predicates=predicates,
                                           target=mapping["target"],
                                           target_domain_metadata=target_domain_metadata,
                                           prefix_path="./experiment_11",
                                           name=mapping_id)

            target_domain_data = read_target_domain(target_domain_path, folds=["fold00"])
            selector_model.train(target_domain_data)

            models_metadata[set_to_uuid(predicates)] = selector_model.get_metadata()
            SelectorModel.update_model_metadata(model_metadata_path_file, models_metadata)
            save_to_pickle(os.path.join(models_path, mapping["target"], set_to_uuid(predicates)),
                           selector_model)
        else:
            # load model
            selector_model = load_pickle(
                os.path.join(models_path, mapping["target"], set_to_uuid(predicates)))

        # read source domain
        source_domain_path = os.path.join(data_path, mapping["source"])

        source_domain_mapped_data = read_source_domain_and_apply_mapping(source_domain_path,
                                                                         mapping,
                                                                         target_domain_metadata)

        selector_model.test(source_domain_mapped_data)

        save_mapped_dataset(source_domain_mapped_data["facts"], os.path.join(base_path, "best", f"{mapping_id}/facts.pl"))
        save_mapped_dataset(source_domain_mapped_data["facts"], os.path.join(base_path, "worst", f"{mapping_id}/facts.pl"))

        scores_by_mapping[mapping_id] = selector_model.load_estimates()

    save_selected_samples(scores_by_mapping=scores_by_mapping, base_path="./experiment_11")

    return transfer_learning_workflow()

def transfer_learning_workflow(target_domain: str, mapped_data_path, model_settings, model_path, train_folds: list = None):
    target_domain_path = f"./preprocessed/{target_domain}"
    mappings_path = f"./mappings/{target_domain}.json"

    with open(mappings_path, "r", encoding="utf-8") as f:
        mappings = json.load(f)

    if train_folds is None:
        train_folds = ["fold00"]

    test_folds = []
    for name in os.listdir(target_domain_path):
        if os.path.isdir(os.path.join(target_domain_path, name)) and name not in train_folds:
            test_folds.append(name)

    test_folds.sort()

    # ------------------------------------------------------------------------------------------------------------------
    # Transforming the datasets in Srlearn databases

    database = DatabaseWrapper()
    database.process_mappings(mappings, mapped_data_path)
    database.process_target(target_domain_path)

    train_database = database.generate_train_srldatabase(train_folds, use_recursion=False)
    test_database = database.generate_test_srldatabase(test_folds, use_recursion=False)

    model_settings = {
        "nEstimators": model_settings.get("nEstimators", 10),
        "nodeSize": model_settings.get("nodeSize", 2),
        "maxTreeDepth": model_settings.get("maxTreeDepth", 3),
        "neg_pos_ratio":model_settings.get("neg_pos_ratio", 2),
        "numberOfClauses": model_settings.get("numberOfClauses", 8),
        "numberOfCycles": model_settings.get("numberOfCycles", 100),
        "focal_loss_alpha": model_settings.get("focal_loss_alpha", 0.66),
        "focal_loss_gamma": model_settings.get("focal_loss_gamma", 2),
        "max_tree_depth_in_literals": model_settings.get("max_tree_depth_in_literals", 3),
        "boxcox_lambda": model_settings.get("boxcox_lambda", 0),
    }

    # ------------------------------------------------------------------------------------------------------------------
    # Training the models
    model = RDNBoost(
        n_estimators=model_settings.get("nEstimators", 10),
        node_size=model_settings.get("nodeSize", 2),
        max_tree_depth=model_settings.get("maxTreeDepth", 3),
        neg_pos_ratio=model_settings.get("neg_pos_ratio", 2),
        number_of_clauses=model_settings.get("numberOfClauses", 8),
        number_of_cycles=model_settings.get("numberOfCycles", 100),
        focal_loss_alpha=model_settings.get("focal_loss_alpha", 0.66),
        focal_loss_gamma=model_settings.get("focal_loss_gamma", 2),
        max_tree_depth_in_literals=model_settings.get("max_tree_depth_in_literals", 3),
        boxcox_lambda=model_settings.get("boxcox_lambda", 0),
        path=model_path
    )

    ignore_stdout = True

    model.fit(train_database, ignoreSTDOUT=ignore_stdout)

    model.predict(train_database, ignoreSTDOUT=ignore_stdout, threshold=0.5)

    train_metrics = extract_metrics_from_log(os.path.join(model_path, "bsrl_data/data0", "test_output.txt"))

    model.predict(test_database, threshold=0.5)

    test_metrics = extract_metrics_from_log(os.path.join(model_path, "bsrl_data/data0", "test_output.txt"))

    return model_settings, train_metrics, test_metrics

def default_workflow(base_path: str, target_domain: str, model_settings, model_path, rounds = 3, small_dataset=True, use_recursion=True):
    target_domain_path = f"./preprocessed/{target_domain}"
    fold_splits = generate_folds(target_domain_path, small_dataset)

    # ------------------------------------------------------------------------------------------------------------------
    # Transforming the datasets in Srlearn databases

    output = []

    for fold_configuration in fold_splits:
        for _ in range(rounds):
            database = DatabaseWrapper()
            database.process_target(target_domain_path)

            train_database = database.generate_train_srldatabase(fold_configuration["train"], use_recursion=use_recursion)
            test_database = database.generate_test_srldatabase(fold_configuration["test"], use_recursion=use_recursion)

            model_settings = {
                "nEstimators": model_settings.get("nEstimators", 10),
                "nodeSize": model_settings.get("nodeSize", 2),
                "maxTreeDepth": model_settings.get("maxTreeDepth", 3),
                "neg_pos_ratio": model_settings.get("neg_pos_ratio", 2),
                "numberOfClauses": model_settings.get("numberOfClauses", 8),
                "numberOfCycles": model_settings.get("numberOfCycles", 100),
                "focal_loss_alpha": model_settings.get("focal_loss_alpha", 1),
                "focal_loss_gamma": model_settings.get("focal_loss_gamma", 2),
                "max_tree_depth_in_literals": model_settings.get("max_tree_depth_in_literals", -1),
                "boxcox_lambda": model_settings.get("boxcox_lambda", 0),
                "rdn_version": model_settings.get("rdn_version", "rdn"),
            }

            # ------------------------------------------------------------------------------------------------------------------
            # Training the models
            model = RDNBoost(
                n_estimators=model_settings.get("nEstimators", 10),
                node_size=model_settings.get("nodeSize", 2),
                max_tree_depth=model_settings.get("maxTreeDepth", 3),
                neg_pos_ratio=model_settings.get("neg_pos_ratio", 2),
                number_of_clauses=model_settings.get("numberOfClauses", 8),
                number_of_cycles=model_settings.get("numberOfCycles", 100),
                focal_loss_alpha=model_settings.get("focal_loss_alpha", 1),
                focal_loss_gamma=model_settings.get("focal_loss_gamma", 2),
                max_tree_depth_in_literals=model_settings.get("max_tree_depth_in_literals", -1),
                boxcox_lambda=model_settings.get("boxcox_lambda", 0),
                rdn_version=model_settings.get("rdn_version", "rdn"),
                path=model_path
            )

            ignore_stdout = True

            model.fit(train_database, ignoreSTDOUT=ignore_stdout)

            model.predict(train_database, ignoreSTDOUT=ignore_stdout, threshold=0.5)

            train_metrics = extract_metrics_from_log(os.path.join(model_path, "bsrl_data/data0", "test_output.txt"))

            model.predict(test_database, threshold=0.5)

            test_metrics = extract_metrics_from_log(os.path.join(model_path, "bsrl_data/data0", "test_output.txt"))

            print(test_metrics)

            output.append({"train_folds": fold_configuration["train"], "test_folds": fold_configuration["test"],  "train": train_metrics, "test": test_metrics})

            shutil.rmtree(os.path.join(base_path, "bsrl_data"))
            #update_experiment_state(os.path.join(base_path, "results.json"), {"train": train_metrics, "test": test_metrics})

    return output

def placeholder(base_path: str, target_domain: str, rounds: int):
    import random

    target_domain_path = f"./preprocessed/{target_domain}"
    mappings_path = os.path.join(base_path, "mappings", f"{target_domain}.json")

    with open(mappings_path, "r", encoding="utf-8") as f:
        mappings = json.load(f)

    for source_domain in []:
        mappings_from = [key for key, value in mappings.items() if value["source"] == source_domain]

        selected_mappings_keys = random.sample(mappings_from, rounds) if rounds > len(mappings_from) else mappings_from

        for mapping_key in selected_mappings_keys:
            curr_mapping = mappings[mapping_key]
            database = DatabaseWrapper()
            database.process_mappings(mappings, mapped_data_path)
            database.process_target(target_domain_path)


    # Dado um par source-target
    # Escolher N mapeamentos do source. Se o total de mapeamentos for menor que N, usar o total. Salvar os mapeamentos escolhidos
    # Para cada mapeamento, aplicar o mapeamento, juntar os datasets e treinar.
    # Avaliar o resultado no conjunto de teste
    # Salvar os resultados

    pass