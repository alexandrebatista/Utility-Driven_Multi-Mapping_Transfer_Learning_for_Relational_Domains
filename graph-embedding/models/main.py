import json
import os

from srlearn.rdn import RDNBoost

from database.database import DatabaseWrapper

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

def models_workflow_2(target_domain: str, mapped_data_path, model_settings, model_path, train_folds: list = None):
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
        boxcox_lambda=model_settings.get("boxcox_lambda", -0.3),
        path=model_path
    )

    ignore_stdout = True

    model.fit(train_database, ignoreSTDOUT=ignore_stdout)

    #model.predict(train_database, ignoreSTDOUT=ignore_stdout, threshold=0.5)

    model.predict(test_database, threshold=0.5)