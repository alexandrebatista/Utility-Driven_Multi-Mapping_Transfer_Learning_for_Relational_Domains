from srlearn import Database
from srlearn.rdn import RDNBoostTransferLearning
from srlearn.weight import WeightFactory


def train(database_train, database_test, settings):
    weight_factory = WeightFactory()
    weight_strategy = weight_factory.getWeightStrategy("scalar", weight=1)

    database = Database.prepareTransferLearningDatabase(
        sourceDatabase=
        database_train,
        weightStrategy=weight_strategy
    )

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
        path=path
    )

    ignore_stdout = True

    # logger.info(f"Training the model on the training set...")
    model.fit(database, ignoreSTDOUT=ignore_stdout)

    # logger.info(f"Evaluating the model on the test set...")
    #model._run_inference(database_test, ignoreSTDOUT=ignore_stdout)

    #model._generate_dotimages()
    #dotImages = model._dotimages
    #metrics = model._prediction_metrics