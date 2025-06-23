import hashlib
import json
import os
import pickle
from uuid import uuid5, UUID

from srlearn.rdn import RDNBoost

from database.database import DatabaseWrapper


def data_to_database(data):
    database = DatabaseWrapper()
    return database.generate_train_srldatabase(data, use_recursion=True)


def set_to_uuid(s):
    namespace = UUID('00000000-0000-0000-0000-000000000000')
    key_base = json.dumps(sorted(s), sort_keys=True).encode('utf-8')
    return uuid5(namespace, hashlib.sha1(key_base).hexdigest())


class SelectorModel:

    def __init__(self, valid_predicates, source, target):
        self._predicates = valid_predicates
        self._model = None
        self._model_name = ""

        self._pkl_path = os.path.join(f"./selector_models/pkl/{target}/")
        self._rdn_boost_path = os.path.join(f"./selector_models/rdn-boost/{target}/")

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
        filtered_data = {

        }

        return filtered_data

    def _check_if_model_exists(self):
        if os.path.exists(path):
            self._load_model()
            pass

    def _load_model(self):
        with open(self._pkl_path, 'rb') as file:
            self._model = pickle.load(file)

    def _save_model(self):
        with open(model_path, 'wb') as file:
            pickle.dump(self._model, file)

    def train(self, domain_data):
        filtered_domain_data = self._filter_predicates(domain_data)
        database = data_to_database(filtered_domain_data)

        self._model.fit(database, ignoreSTDOUT=self._settings["ignoreSTDOUT"])

    def test(self, data):
        self._model.predict(data, threshold=0.5)
        self._extract_test_metrics()

    def _extract_train_metrics(self):
        pass

    def _extract_test_metrics(self):
        pass

