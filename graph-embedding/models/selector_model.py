import os
import pickle

from srlearn.rdn import RDNBoost

from database.utils import graph_data_to_database
from models.utils import set_to_uuid, extract_metrics_from_log, extract_learning_time_from_log


class SelectorModel:

    def __init__(self, valid_predicates, source, target, target_domain_metadata):
        self._name = set_to_uuid(valid_predicates)
        self._predicates = valid_predicates
        self._model = None

        self._train_metrics = {}
        self._test_metrics = {}

        self._target_domain_metadata = target_domain_metadata

        self._pkl_path = os.path.join(f"./selector_models/pkl/{target}/")
        self._rdn_boost_path = os.path.join(f"./selector_models/rdn-boost/{target}/{self._name}")
        self._train_output_path = os.path.join(self._rdn_boost_path, "bsrl_data/data0", "train_output.txt")
        self._test_output_path = os.path.join(self._rdn_boost_path, "bsrl_data/data0", "test_output.txt")

        self._settings = {
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

        self._initialize_model()

    def _initialize_model(self):
        self._model = RDNBoost(
            n_estimators=self._settings["nEstimators"],
            node_size=self._settings["nodeSize"],
            max_tree_depth=self._settings["maxTreeDepth"],
            neg_pos_ratio=self._settings["neg_pos_ratio"],
            number_of_clauses=self._settings["numberOfClauses"],
            number_of_cycles=self._settings["numberOfCycles"],
            path=self._rdn_boost_path
        )


    def _filter_predicates(self, data):
        all_predicates = list(data["facts"])

        for predicate in all_predicates:
            if predicate not in self._predicates and predicate != self._target_domain_metadata["target_predicate"]:
                del data["facts"][predicate]

        return data

    def train(self, domain_data):
        self._filter_predicates(domain_data)
        database = graph_data_to_database(domain_data, self._target_domain_metadata)

        self._model.fit(database, ignoreSTDOUT=self._settings["ignoreSTDOUT"])
        self._model.predict(database, threshold=0.5)
        self._extract_train_metrics()

    def test(self, data):
        database = graph_data_to_database(data, self._target_domain_metadata)

        self._test_metrics = {}
        self._model.predict(database, threshold=0.5)
        self._extract_test_metrics()

    def _extract_train_metrics(self):
        self._train_metrics = extract_metrics_from_log(self._test_output_path)
        self._train_metrics["learning_time_s"] = extract_learning_time_from_log(self._train_output_path)

    def _extract_test_metrics(self):
        self._test_metrics = extract_metrics_from_log(self._test_output_path)

    def get_train_metrics(self):
        return self._train_metrics

    def get_test_metrics(self):
        return self._test_metrics

    def get_metadata(self):
        return {
            "name": self._name,
            "predicates": list(self._predicates),
            "train_metrics": self._train_metrics
        }
