import json
import os
import shutil

from srlearn.rdn import RDNBoost

from database.database import DatabaseWrapper

from models.utils import extract_metrics_from_log, create_folder, create_folder, set_to_uuid, load_pickle, save_to_pickle
from .selector_model import SelectorModel
from .utils import create_folder, set_to_uuid, load_pickle, save_to_pickle, generate_folds
from models.model import generate_knowledge_gap_data
from models.selector_model import get_transfer_dataset
from database.utils import extract_domain_metadata, read_target_domain
from mapping.apply_mappings import read_source_domain_and_apply_mapping
from mapping.utils import select_mappings


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

# ======================================================================================================================

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

def tl_experiment_1(base_path: str, model_path, target_domain: str, rounds: int, use_recursion: bool = True):
    data_path = f"./preprocessed/"
    target_domain_path = os.path.join(data_path, target_domain)

    database = DatabaseWrapper()
    database.process_target(target_domain_path)

    mappings_path = os.path.join(base_path, "mappings", f"{target_domain}.json")

    with open(mappings_path, "r", encoding="utf-8") as f:
        mappings = json.load(f)

    for source_domain in []:
        if source_domain == target_domain:
            continue

        source_domain_path = os.path.join(data_path, source_domain)

        selected_mappings = select_mappings(mappings, source_domain, rounds)

        for curr_mapping_key, curr_mapping in selected_mappings.items():
            source_domain_mapped_data = read_source_domain_and_apply_mapping(
                source_domain_path,
                curr_mapping,
                database.get_target_domain_metadata()
            )

            database.process_source_from_data(source_domain_mapped_data)

            train_metrics, test_metrics = training_workflow(
                base_path=base_path,
                model_path=model_path,
                database=database,
                model_settings={},
                use_recursion=use_recursion
            )

            with open(os.path.join(base_path, "default_approach_results.json"), 'w') as file:
                json.dump(
                    {"train_results": train_metrics, "test_results": test_metrics},
                    file
                )

            train_metrics, test_metrics = kg_workflow(
                base_path=base_path,
                source_domain_mapped_data=source_domain_mapped_data,
                target_domain=target_domain,
                model_settings={},
                use_recursion=use_recursion,
                rounds=rounds
            )

            with open(os.path.join(base_path, "kg_approach_results.json"), 'w') as file:
                json.dump(
                    {"train_results": train_metrics, "test_results": test_metrics},
                    file
                )

    # Avaliar o resultado no conjunto de teste
    # Salvar os resultados
    # Dado um par source-target
    # Escolher N mapeamentos do source. Se o total de mapeamentos for menor que N, usar o total. Salvar os mapeamentos escolhidos
    # Para cada mapeamento, aplicar o mapeamento, juntar os datasets e treinar.

# ======================================================================================================================

# Knowledge Gap Functions

def kg_workflow(base_path, source_domain_mapped_data, target_domain, model_settings, use_recursion: bool, rounds: int=3):
    target_domain_path = f"./preprocessed/{target_domain}"

    target_dataset = DatabaseWrapper()
    target_dataset.process_target(target_domain_path)
    fold_splits = target_dataset.generate_fold_splits()

    tl_train_metrics_list = []
    tl_test_metrics_list = []

    for fold_configuration in fold_splits:

        train_target_dataset = target_dataset.generate_train_srldatabase(
            fold_configuration["train"],
            use_recursion=use_recursion
        )

        test_target_dataset = target_dataset.generate_test_srldatabase(
            fold_configuration["test"],
            use_recursion=use_recursion
        )

        for _ in range(rounds):
            # Generate dataset with misclassified by the model
            knowledge_gap_data = generate_knowledge_gap_data(
                dataset = target_dataset,
                fold_configuration = fold_configuration,
                use_recursion=use_recursion
            )

            # Get transfer dataset
            transfer_learning_raw_dataset = get_transfer_dataset(source_domain_mapped_data, knowledge_gap_data)

            tl_dataset = DatabaseWrapper()
            tl_dataset.process_source_from_data(transfer_learning_raw_dataset)
            tl_dataset.process_target(target_domain_path)

            transfer_learning_train_dataset = tl_dataset.generate_train_srldatabase(
                fold_configuration["train"],
                use_recursion=use_recursion
            )

            tl_train_metrics, tl_test_metrics = kg_training_workflow(
                base_path=base_path,
                train_dataset=transfer_learning_train_dataset,
                test_dataset=test_target_dataset,
                model_settings=model_settings,
            )

            tl_train_metrics_list.append(tl_train_metrics)
            tl_test_metrics_list.append(tl_test_metrics)

    return tl_train_metrics_list, tl_test_metrics_list

def training_workflow(base_path, model_path, database, use_recursion, model_settings=None, rounds=3):
    if model_settings is None:
        model_settings = {}

    train_metrics_list = []
    test_metrics_list = []

    fold_splits = database.generate_fold_splits()

    for fold_configuration in fold_splits:
        for _ in range(rounds):
            train_database = database.generate_train_srldatabase(fold_configuration["train"], use_recursion=use_recursion)
            test_database = database.generate_test_srldatabase(fold_configuration["test"], use_recursion=use_recursion)

            model = RDNBoost(**model_settings)

            ignore_stdout = True
            model.fit(train_database, ignoreSTDOUT=ignore_stdout)

            model.predict(train_database, ignoreSTDOUT=ignore_stdout, threshold=0.5)
            train_metrics = extract_metrics_from_log(os.path.join(model_path, "bsrl_data/data0", "test_output.txt"))

            model.predict(test_database, threshold=0.5)
            test_metrics = extract_metrics_from_log(os.path.join(model_path, "bsrl_data/data0", "test_output.txt"))

            shutil.rmtree(os.path.join(base_path, "bsrl_data"))

            train_metrics_list.append(train_metrics)
            test_metrics_list.append(test_metrics)

    return train_metrics_list, test_metrics_list

def kg_training_workflow(base_path, train_dataset, test_dataset, model_settings=None):
    if model_settings is None:
        model_settings = {}

    model = RDNBoost(**model_settings)

    ignore_stdout = True
    model.fit(train_dataset, ignoreSTDOUT=ignore_stdout)
    model.predict(train_dataset, ignoreSTDOUT=ignore_stdout, threshold=0.5)
    train_metrics = extract_metrics_from_log(os.path.join(base_path, "bsrl_data/data0", "test_output.txt"))

    model.predict(test_dataset, ignoreSTDOUT=ignore_stdout, threshold=0.5)
    test_metrics = extract_metrics_from_log(os.path.join(base_path, "bsrl_data/data0", "test_output.txt"))

    return train_metrics, test_metrics
