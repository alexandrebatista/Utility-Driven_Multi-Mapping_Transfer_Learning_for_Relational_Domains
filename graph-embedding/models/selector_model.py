import os

from srlearn.rdn import RDNBoost

from database.utils import graph_data_to_database, raw_data_to_database
from models.utils import set_to_uuid, extract_metrics_from_log, extract_learning_time_from_log


class SelectorModel:

    def __init__(self, valid_predicates, source, target, target_domain_metadata, settings=None, rdn_boost_path=None):
        self._name = set_to_uuid(valid_predicates)
        self._predicates = valid_predicates
        self._model = None

        self._train_metrics = {}
        self._test_metrics = {}

        self._target_domain_metadata = target_domain_metadata

        self._pkl_path = os.path.join(f"./selector_models/pkl/{target}/")
        self._rdn_boost_path = os.path.join(
            f"./selector_models/rdn-boost/{target}/{self._name}") if rdn_boost_path is None else rdn_boost_path
        self._train_output_path = os.path.join(self._rdn_boost_path, "bsrl_data/data0", "train_output.txt")
        self._test_output_path = os.path.join(self._rdn_boost_path, "bsrl_data/data0", "test_output.txt")

        if settings is not None:
            self._settings = settings
        else:
            self._settings = {
                "randomSeed": 11,
                "numberOfClauses": 8,
                "numberOfCycles": 100,
                "maxTreeDepth": 3,
                "nEstimators": 100,
                "nodeSize": 2,
                "neg_pos_ratio": 2,
                "max_failed_neg_sampling_retries": 50,
                "ignoreSTDOUT": True,
                "useRecursion": True,
                "runOriginalRDNBoost": False,
                "no_boost": False,
                "debug": False
            }

        print(self._settings)

        self._initialize_model()

    def _initialize_model(self):
        self._model = RDNBoost(
            n_estimators=self._settings["nEstimators"],
            node_size=self._settings["nodeSize"],
            max_tree_depth=self._settings["maxTreeDepth"],
            neg_pos_ratio=self._settings["neg_pos_ratio"],
            number_of_clauses=self._settings["numberOfClauses"],
            number_of_cycles=self._settings["numberOfCycles"],
            path=self._rdn_boost_path,
            no_boost=self._settings["no_boost"],
            debug=self._settings["debug"]
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
        database = raw_data_to_database(data, self._target_domain_metadata)

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

    def get_best_samples(self):
        pass

    def load_estimates(self):
        """
        Loads predicate estimates from a file and returns a list of dictionaries with:
        - 'predicate': the (possibly negated) predicate name
        - 'args': list of arguments
        - 'raw': the raw predicate string
        - 'score': the float score at the end of the line
        """
        import re

        estimates = []
        pattern = re.compile(r"^(!?\w+)\(([^)]*)\)\s+([0-9]*[.,]?[0-9]+)$")

        filepath = os.path.join(self._rdn_boost_path, "bsrl_data", "data0", "test",
                                f"results_{self._target_domain_metadata['target_predicate']}.db")

        print(filepath)

        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("%"):
                    continue

                match = pattern.match(line)
                if not match:
                    print(f"Skipping malformed line: {line}")
                    continue

                predicate = match.group(1)
                args_str = match.group(2)
                args = args_str.split(",")
                score_str = match.group(3).replace(",", ".")  # in case commas are used as decimal separator

                try:
                    score = float(score_str)
                except ValueError:
                    print(f"Could not convert score to float in line: {line}")
                    continue

                estimates.append({
                    "predicate": predicate,
                    "args": args,
                    "raw": f"{predicate}({','.join(args)})",
                    "score": score
                })

        return estimates

    def get_estimates_by_score(self, threshold=0.5):
        estimates = self.load_estimates()
        return [item for item in estimates if item.get('score', 1) <= threshold]

    