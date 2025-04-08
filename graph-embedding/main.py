from pathlib import Path

from srlearn.rdn import RDNBoostTransferLearning

from database.database import DatabaseWrapper
from mapping.mapping import generate_mappings
from mapping.apply_mappings import apply_mappings
from mapping.select_mappings import select_mappings
#from train.train_model import train


def create_folder(path):
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

# Aplicar a metodologia da Thaís se o número de mapeamentos for maior que 10 (?)

def generating_mappings_for(target_domain_name):
    base_dir = "./preprocessed"
    target_directory = f"{base_dir}/{target_domain_name}/fold00"
    output_path = f"/home/alexandre/mestrado/graph-embedding/mapped_datasets/to_{target_domain_name}"

    create_folder(output_path)

    print(f"to {target_domain_name}")

    path = Path(base_dir)
    for item in path.iterdir():
        if item.is_dir() and item.name != target_domain_name:
            source_path = os.path.join(item)
            source_directory = f"{source_path}/fold00"

            solutions = generate_mappings(source_directory, target_directory)
            #solutions.print_coherent_mappings()

            print(f"\t {item.name} : {len(solutions.get_robust_mappings())}")

            create_folder(os.path.join(output_path, item.name))

            mappings_subset = select_mappings(solutions)

            apply_mappings(source_path,
                           os.path.join(output_path, item.name),
                           mappings_subset,
                           solutions.get_target_target_predicate(),
                           solutions.get_target_modes())


# def train_for(target_domain_name):
#     settings = {
#         "randomSeed": 11,
#         "numberOfClauses": 8,
#         "numberOfCycles": 100,
#         "maxTreeDepth": 3,
#         "nEstimators": 10,
#         "nodeSize": 2,
#         "neg_pos_ratio": 2,
#         "max_failed_neg_sampling_retries": 50,
#         "ignoreSTDOUT": True,
#         "useRecursion": False,
#         "runOriginalRDNBoost": True
#     }
#
#     base_dir = "./preprocessed"
#     target_directory = f"{base_dir}/{target_domain_name}/fold00"
#
#     database_train = None
#     database_test = None
#
#     target_database_train = load_database(
#         use_recursion=settings["useRecursion"],
#         logger=logger,
#         neg_pos_ratio=settings["neg_pos_ratio"],
#         max_failed_neg_sampling_retries=settings["max_failed_neg_sampling_retries"],
#     )
#
#     train(database_train, database_test, settings)


if __name__ == '__main__':
    import os

    os.nice(10)

    # ------------------------------------------------------------------------------------------------------------------
    # Generating Mappings and saving files

    path = Path("./preprocessed")
    generating_mappings_for("cora")
    # generating_mappings_for("imdb")
    # generating_mappings_for("nell_finances")
    # generating_mappings_for("nell_sports")
    # generating_mappings_for("twitter")
    # generating_mappings_for("uwcse")
    # generating_mappings_for("yeast")

    # ------------------------------------------------------------------------------------------------------------------
    # Transforming the datasets in Srlearn databases

    # database = DatabaseWrapper()
    # database.process_source("./mapped_datasets/to_cora")
    # database.process_target("./preprocessed/cora")
    #
    # train_database = database.generate_train_srldatabase(["fold00", "fold01", "fold02", "fold03"], use_recursion=True)
    # test_database = database.generate_test_srldatabase(["fold04"], use_recursion=False)
    #
    # # ------------------------------------------------------------------------------------------------------------------
    # # Training the models
    #
    # settings = {
    #             "randomSeed": 11,
    #             "numberOfClauses": 8,
    #             "numberOfCycles": 100,
    #             "maxTreeDepth": 3,
    #             "nEstimators": 10,
    #             "nodeSize": 2,
    #             "neg_pos_ratio": 2,
    #             "max_failed_neg_sampling_retries": 50,
    #             "ignoreSTDOUT": True,
    #             "useRecursion": False,
    #             "runOriginalRDNBoost": True
    #         }
    #
    # db = database.generate_tl_database(["fold00", "fold01", "fold02", "fold03"], use_recursion=True)
    # utility_alpha = 1
    # model = RDNBoostTransferLearning(
    #     n_estimators=settings["nEstimators"],
    #     node_size=settings["nodeSize"],
    #     max_tree_depth=settings["maxTreeDepth"],
    #     neg_pos_ratio=settings["neg_pos_ratio"],
    #     number_of_clauses=settings["numberOfClauses"],
    #     number_of_cycles=settings["numberOfCycles"],
    #     source_utility_alpha=utility_alpha,
    #     target_utility_alpha=utility_alpha,
    #     path="./"
    # )
    #
    # ignore_stdout = True
    #
    # # logger.info(f"Training the model on the training set...")
    # model.fit(db, ignoreSTDOUT=ignore_stdout)
    #
    # pass
    #
