import os

from srlearn.rdn import RDNBoost

from database.utils import graph_data_to_database, raw_data_to_database
from models.utils import set_to_uuid, extract_metrics_from_log, extract_learning_time_from_log, create_folder

def load_model(pickle_path):
    import pickle

    with open(pickle_path, 'rb') as file:
        loaded_object = pickle.load(file)

    return loaded_object

class RdnBoost:

    def __init__(self, target_domain_metadata, model_settings=None, prefix_path="./", name=None):
        self._model = None
        self._name = name

        self._train_metrics = {}
        self._test_metrics = {}

        self._target_domain_metadata = target_domain_metadata

        self._pkl_path = os.path.join(prefix_path, f"selector_models/pkl/")
        self._rdn_boost_path = os.path.join(prefix_path, f"selector_models/rdn-boost/{self._name}")
        self._train_output_path = os.path.join(self._rdn_boost_path, "bsrl_data/data0", "train_output.txt")
        self._test_output_path = os.path.join(self._rdn_boost_path, "bsrl_data/data0", "test_output.txt")

        self._model_settings = model_settings if model_settings is not None else {}

        print(self._model_settings)

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

    def train(self, domain_data):
        #database = graph_data_to_database(domain_data, self._target_domain_metadata)

        self._model.fit(domain_data, ignoreSTDOUT=True)
        self._model.predict(domain_data, threshold=0.5)
        self._extract_train_metrics()

    def test(self, data):
        #database = raw_data_to_database(data, self._target_domain_metadata)
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
                    "raw": f"{predicate}({','.join(args)}).",
                    "score": score
                })

        return estimates

    def get_estimates_by_score(self, threshold=0.5):
        estimates = self.load_estimates()
        return [item for item in estimates if item.get('score', 1) <= threshold]

    def get_train_accuracies(self, threshold):
        import numpy as np

        scores = self.load_estimates()

        # Extract labels and scores as NumPy arrays
        y_true = np.array([0 if s["raw"].startswith("!") else 1 for s in scores])
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

    def save(self):
        """
        Saves itself to a pickle file at the specified filepath.

        """
        import pickle

        create_folder(self._pkl_path)

        with open(os.path.join(self._pkl_path, f"{self._name}.pkl"), 'wb') as file:
            pickle.dump(self, file)

    def delete(self):
        import shutil

        pickle_file_path = os.path.join(self._pkl_path, f"{self._name}.pkl")

        if os.path.exists(pickle_file_path):
            os.remove(pickle_file_path)
            print(f"File '{pickle_file_path}' deleted successfully.")
        else:
            print(f"File '{pickle_file_path}' does not exist.")

        if os.path.exists(self._rdn_boost_path) and os.path.isdir(self._rdn_boost_path):
            shutil.rmtree(self._rdn_boost_path)
            print(f"Folder '{self._rdn_boost_path}' deleted successfully.")
        else:
            print(f"Folder '{self._rdn_boost_path}' does not exist.")

    def process_data(self, min_fraction=0.01, score_threshold=0.5):
        from collections import defaultdict

        # --- separar positivos e negativos ---
        def split_by_label(data):
            pos = [d for d in data if not d["raw"].startswith("!")]
            neg = [{**d, "raw": d["raw"][1:]} for d in data if d["raw"].startswith("!")]
            return pos, neg

        # --- remover scores raros (menos de min_fraction do grupo) ---
        def remove_rare_scores(instances, min_fract=0.01):
            freq = defaultdict(int)
            total = len(instances)

            for d in instances:
                freq[d["score"]] += 1

            threshold = total * min_fract
            filtered = [d for d in instances if freq[d["score"]] >= threshold]
            return filtered

        # --- dividir por score >= threshold ou < threshold ---
        def split_by_score_threshold(instances, threshold=0.5):
            above = [d for d in instances if d["score"] >= threshold]
            below = [d for d in instances if d["score"] < threshold]
            return above, below

        def save_datasets(path, pos, neg):
            with open(os.path.join(path, 'pos.pl'), 'w') as f:
                for index, item in enumerate(pos):
                    raw_clean = item['raw'].replace(" ", "") + "."
                    f.write(raw_clean + "\n")
                    if index < len(pos) - 1:
                        f.write("\n")

            with open(os.path.join(path, 'neg.pl'), 'w') as f:
                for index, item in enumerate(neg):
                    raw_clean = item['raw'].replace(" ", "") + "."
                    f.write(raw_clean)
                    if index < len(neg) - 1:
                        f.write("\n")

        estimates = self.load_estimates()

        positives, negatives = split_by_label(estimates)

        # remover anomalias
        pos_filtered = remove_rare_scores(positives, min_fraction)
        neg_filtered = remove_rare_scores(negatives, min_fraction)

        # dividir por threshold de score
        pos_above, pos_below = split_by_score_threshold(pos_filtered, score_threshold)
        neg_above, neg_below = split_by_score_threshold(neg_filtered, score_threshold)

        #save_datasets(self._above_path, pos_above, neg_above)
        #save_datasets(self._below_path, pos_below, neg_below)

        return {
            "pos_above": pos_above,
            "pos_below": pos_below,
            "neg_above": neg_above,
            "neg_below": neg_below
        }


def search_best_model(data, metadata, prefix_path, save=False):
    model_settings = {}

    number_trees_list = [10, 20, 50, 100]
    max_tree_depth_in_literals_list = [3, 4, 5, 6, 7, 8, 9, 10]

    for number_trees in number_trees_list:
        model_settings["nEstimators"] = number_trees

        for max_tree_depth_in_literals in max_tree_depth_in_literals_list:
            model_settings["max_tree_depth_in_literals"] = max_tree_depth_in_literals

            model = RdnBoost(target_domain_metadata=metadata,
                             model_settings=model_settings,
                             prefix_path=prefix_path,
                             name=f"{str(number_trees)}_{str(max_tree_depth_in_literals)}")

            model.train(data)

            pos_acc, neg_acc = model.get_train_accuracies(0.5)

            print(
                f"{str(number_trees)}:{str(max_tree_depth_in_literals)} - Positive Accuracy: {str(pos_acc)}. Negative Accuracy: {str(neg_acc)}")

            if pos_acc > 0.5 and neg_acc > 0.5:
                if save:
                    model.save()
                print("Model found!")
                print(number_trees)
                print(max_tree_depth_in_literals)
                return model

    raise Exception("A good model, to generate the knowledge gap dataset, could not be found.")

def generate_knowledge_gap_data(dataset, fold_configuration, use_recursion, base_path):
    train_dataset = dataset.generate_train_srldatabase(
        fold_configuration["train"],
        use_recursion=use_recursion
    )

    target_domain_metadata = dataset.get_target_domain_metadata()
    best_model = search_best_model(train_dataset, target_domain_metadata, base_path)
    filtered_samples = best_model.process_data(min_fraction=0.01, score_threshold=0.5)
    best_model.delete()

    knowledge_gap_samples = {
        "pos": [item["raw"] for item in filtered_samples["pos_below"]],
        "neg": [item["raw"] for item in filtered_samples["neg_below"]],
    }

    knowledge_gap_data = {
        **dataset.get_target_facts(fold_configuration["train"]), **knowledge_gap_samples
    }

    return knowledge_gap_data
