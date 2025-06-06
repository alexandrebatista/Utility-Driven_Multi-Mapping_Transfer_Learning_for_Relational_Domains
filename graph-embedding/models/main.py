import os

from srlearn.rdn import RDNBoostTransferLearning

from database.database import DatabaseWrapper

def models_workflow(target_domain: str, train_folds: list = None, source_domains: list = None, quality: str = "best"):
    target_domain_path = f"./preprocessed/{target_domain}"

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
    database.process_source(source_domains)
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
    model = RDNBoostTransferLearning(
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

    # logger.info(f"Training the model on the training set...")
    model.fit(train_database, ignoreSTDOUT=ignore_stdout)

    model.predict(test_database, threshold=0.5)

    # model._prediction_metrics