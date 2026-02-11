import os
import pickle

from srlearn.rdn import RDNBoost

from database.utils import graph_data_to_database, raw_data_to_database, convert_to_prolog_data, convert_datatomap_to_prologdata
from .utils import set_to_uuid, extract_metrics_from_log, extract_learning_time_from_log


class SelectorModel:

    def __init__(self, valid_predicates, target, target_domain_metadata, model_settings=None, prefix_path="./", name=None):
        self._name = set_to_uuid(valid_predicates) if name is None else name
        self._predicates = valid_predicates
        self._model = None

        self._train_metrics = {}
        self._test_metrics = {}

        self._target_domain_metadata = target_domain_metadata

        self._pkl_path = os.path.join(prefix_path, f"selector_models/pkl/{target}/{self._name}.pkl")
        self._rdn_boost_path = os.path.join(prefix_path, f"selector_models/rdn-boost/{target}/{self._name}")
        self._train_output_path = os.path.join(self._rdn_boost_path, "bsrl_data/data0", "train_output.txt")
        self._test_output_path = os.path.join(self._rdn_boost_path, "bsrl_data/data0", "test_output.txt")

        self._model_settings = model_settings if model_settings is not None else {}

        print(model_settings)

        self._initialize_model()

    def _initialize_model(self):
        self._model = RDNBoost(
            n_estimators=self._model_settings.get("nEstimators", 10),
            node_size=self._model_settings.get("nodeSize", 2),
            max_tree_depth=self._model_settings.get("maxTreeDepth", 3),
            neg_pos_ratio=self._model_settings.get("neg_pos_ratio", 2),
            number_of_clauses=self._model_settings.get("numberOfClauses", 8),
            number_of_cycles=self._model_settings.get("numberOfCycles", 100),
            focal_loss_alpha=self._model_settings.get("focal_loss_alpha", 1),
            focal_loss_gamma=self._model_settings.get("focal_loss_gamma", 0),
            max_tree_depth_in_literals=self._model_settings.get("max_tree_depth_in_literals", -1),
            boxcox_lambda=self._model_settings.get("boxcox_lambda", 0),
            rdn_version=self._model_settings.get("rdn_version", "rdn"),
            path=self._rdn_boost_path
    )

    def _filter_predicates(self, data):
        data["facts"] = [
            fact for fact in data["facts"]
            if fact.split('(', 1)[0] in self._predicates and fact.split('(', 1)[0] != self._target_domain_metadata["target_predicate"]
        ]
        return data

    def train(self, domain_data, filter_predicates=True):
        if filter_predicates:
            domain_data = self._filter_predicates(domain_data)

        database = raw_data_to_database(domain_data, self._target_domain_metadata)

        self._model.fit(database, ignoreSTDOUT=True)
        self._model.predict(database, threshold=0.5)
        self._extract_train_metrics()

    def test(self, data):
        database = raw_data_to_database(data, self._target_domain_metadata)
        #database = graph_data_to_database(data, self._target_domain_metadata)

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

    def get_best_samples(self, threshold=0.5):
        def split_by_label(data):
            pos = [d for d in data if not d["negative"]]
            neg = [d for d in data if d["negative"]]
            return pos, neg

        estimates = self.load_estimates()
        best_samples = [item for item in estimates if item.get('score', -1) > threshold]
        best_pos, best_neg = split_by_label(best_samples)
        return best_pos, best_neg

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
                    "predicate": predicate[1:],
                    "args": args,
                    "negative": predicate[0] == "!",
                    "raw": f"{predicate[1:] if predicate[0] == '!' else predicate}({','.join(args)}).",
                    "score": score
                })

        return estimates

    def get_estimates_by_score(self, threshold=0.5):
        estimates = self.load_estimates()
        return [item for item in estimates if item.get('score', 1) <= threshold]

    def get_test_accuracies(self, threshold=0.5):
        import numpy as np

        scores = self.load_estimates()

        # Extract labels and scores as NumPy arrays
        y_true = np.array([0 if s["negative"] else 1 for s in scores])
        score_arr = np.array([s["score"] for s in scores])

        # Predicted = correct if score > threshold, else inverted
        y_pred = np.where(score_arr > threshold, y_true, 1 - y_true)

        # Positive accuracy (class 1)
        pos_mask = y_true == 1
        pos_acc = np.sum(y_pred[pos_mask] == 1) / np.sum(pos_mask) if np.sum(pos_mask) > 0 else 0

        # Negative accuracy (class 0)
        neg_mask = y_true == 0
        neg_acc = np.sum(y_pred[neg_mask] == 0) / np.sum(neg_mask) if np.sum(neg_mask) > 0 else 0

        return float(pos_acc), float(neg_acc)

    def is_reliable(self, threshold=0.5):
        pos_acc, neg_acc = self.get_test_accuracies()
        return pos_acc > threshold and neg_acc > threshold

    def save(self):
        """
        Saves itself to a pickle file at the specified filepath.

        """
        import pickle

        # ensure containing directory exists
        os.makedirs(os.path.dirname(self._pkl_path), exist_ok=True)

        with open(self._pkl_path, 'wb') as file:
            pickle.dump(self, file)

    def delete(self):
        import os

        # remove only the pickle file
        if os.path.exists(self._pkl_path) and os.path.isfile(self._pkl_path):
            os.remove(self._pkl_path)
            print(f"File '{self._pkl_path}' deleted successfully.")
        else:
            print(f"File '{self._pkl_path}' does not exist.")

        if os.path.exists(self._rdn_boost_path) and os.path.isdir(self._rdn_boost_path):
            import shutil
            shutil.rmtree(self._rdn_boost_path)
            print(f"Folder '{self._rdn_boost_path}' deleted successfully.")
        else:
            print(f"Folder '{self._rdn_boost_path}' does not exist.")

    @staticmethod
    def get_models_metadata(metadata_path):
        import json

        if not os.path.exists(metadata_path):
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump({}, f)

            return {}

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        return metadata

    @staticmethod
    def search_model_metadata(metadata, predicates):
        model_uuid = set_to_uuid(predicates)

        return metadata.get(model_uuid)

    @staticmethod
    def update_model_metadata(metadata_path, metadata):
        import json

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)

    @staticmethod
    def load_selector_model(base_path, target_domain_name, valid_predicates):
        expected_pkl_dir = os.path.join(base_path, f"selector_models/pkl/{target_domain_name}/")
        expected_pkl_file = os.path.join(expected_pkl_dir, f"{set_to_uuid(valid_predicates)}.pkl")
        if os.path.exists(expected_pkl_file):
            with open(expected_pkl_file, 'rb') as f:
                selector_model = pickle.load(f)
            return selector_model
        return None


def get_transfer_dataset(base_path: str, source_domain_mapped_data, target_domain_name: str, target_domain_metadata, knowledge_gap_data):
    valid_predicates = list(source_domain_mapped_data["facts"].keys()) + list(source_domain_mapped_data["pos"].keys())

    selector_model = SelectorModel(valid_predicates, target=target_domain_name, target_domain_metadata=target_domain_metadata, prefix_path=base_path)

    # Learn the knowledge gap patterns
    selector_model.train(knowledge_gap_data)

    # Look for source domain samples whose patterns match knowledge gap ones.
    source_domain_converted_data = convert_datatomap_to_prologdata(source_domain_mapped_data)
    selector_model.test(source_domain_converted_data)

    pos_transfer_samples, neg_transfer_samples = selector_model.get_best_samples()

    raw_transfer_dataset = {
        "facts": convert_to_prolog_data(source_domain_mapped_data["facts"]),
        "pos": [f"{sample['raw']}." for sample in pos_transfer_samples],
        "neg": [f"{sample['raw'][1:]}." for sample in neg_transfer_samples]
    }

    selector_model.delete()

    return raw_transfer_dataset

def generate_selector_model(
        base_path: str,
        valid_predicates: list,
        target_domain_name: str,
        target_domain_metadata: dict,
        dataset,
        save=True
):
    # try to load an existing selector model built for these predicates
    selector_model = SelectorModel.load_selector_model(base_path, target_domain_name, valid_predicates)
    if selector_model is not None:
        return selector_model

    number_trees_list = [10, 50, 100]
    max_tree_depth_list = [3, 5, 8, 10]

    for number_trees in number_trees_list:
        for max_tree_depth in max_tree_depth_list:
            model_settings = {
                "disable_limiters": True,
                "nEstimators": number_trees,
                "maxTreeDepth": max_tree_depth,
            }

            selector_model = SelectorModel(
                valid_predicates=valid_predicates,
                model_settings=model_settings,
                prefix_path=base_path,
                target=target_domain_name,
                target_domain_metadata=target_domain_metadata,
            )

            selector_model.train(dataset)

            if selector_model.is_reliable(threshold=0.5):
                if save:
                    selector_model.save()
                print("Model found! Number of trees: " + str(number_trees) + ". Max tree depth: " + str(max_tree_depth))

                return selector_model

            print("Model not found! Number of trees: " + str(number_trees) + ". Max tree depth: " + str(max_tree_depth))
            selector_model.delete()

    return None