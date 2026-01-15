import os
from srlearn import Database
from srlearn.weight import WeightFactory

from models.utils import generate_folds

from .models import DomainMetadata

from .utils import extract_domain_metadata

class DatabaseWrapper:
    def __init__(self):
        self._modes: list = []

        self._target_predicate: str | None = None

        self._target_domain_path: str | None = None

        self._target_domain_metadata: DomainMetadata | None = None

        self._data = {
            "source": {

            },
            "target": {

            }
        }

    def process_source(self, source_domains: list):
        for source_domain_obj in source_domains:
            source_domain = source_domain_obj["source_domain"]
            mappings = source_domain_obj["mappings"]
            self._data["source"][source_domain] = {}

            for mapping_id, mapping_path in mappings.items():
                if os.path.isdir(mapping_path):
                    self._data["source"][source_domain][mapping_id] = {"facts": [], "pos": [], "neg": []}

                    for file in os.listdir(mapping_path):
                        with open(os.path.join(mapping_path, file), "r", encoding="utf-8") as f:
                            lines = f.readlines()
                            self._data["source"][source_domain][mapping_id][file[:-3]].extend([line.strip() for line in lines])

    def reset_source_data(self):
        self._data["source"] = {}

    def process_source_from_data(self, multiple_domain_data):
        for domain, mappings in multiple_domain_data.items():
            for mapping_id, mapping in mappings.items():
                for data_type, data in mapping.items():
                    if domain not in self._data["source"]:
                        self._data["source"][domain] = {}
                    if mapping_id not in self._data["source"][domain]:
                        self._data["source"][domain][mapping_id] = {"facts": [], "pos": [], "neg": []}

                    self._data["source"][domain][mapping_id][data_type].extend(data)

    def process_target_from_data(self, target_data, fold):
        self._data["target"] = {
            fold: {
                "facts": target_data["facts"],
                "pos": target_data["pos"],
                "neg": target_data["neg"]
            }
        }

    def process_mappings(self, mappings, mappings_data_path):
        for mapping_id in os.listdir(mappings_data_path):
            domain = mappings[mapping_id]["source"]

            if domain not in self._data["source"]:
                self._data["source"][domain] = {}
            if mapping_id not in self._data["source"][domain]:
                self._data["source"][domain][mapping_id] = {"facts": [], "pos": [], "neg": []}

            facts_path = os.path.join(mappings_data_path, mapping_id, "facts.pl")
            pos_path = os.path.join(mappings_data_path, mapping_id, "pos.pl")
            neg_path = os.path.join(mappings_data_path, mapping_id, "neg.pl")

            with open(facts_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                self._data["source"][domain][mapping_id]["facts"] = [line.strip() for line in lines]

            if os.path.exists(pos_path):
                with open(pos_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    self._data["source"][domain][mapping_id]["pos"] = [line.strip() for line in lines]

            if os.path.exists(neg_path):
                with open(neg_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    self._data["source"][domain][mapping_id]["neg"] = [line.strip() for line in lines]

    def process_target(self, target_domain_path: str):
        self._target_domain_path = target_domain_path

        self._target_domain_metadata = extract_domain_metadata(target_domain_path)

        for fold in os.listdir(target_domain_path):
            if not self._modes:
                with open(os.path.join(target_domain_path, fold, "modes.pl"), "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    self._modes.extend([line.strip() for line in lines])

            self._data["target"][fold] = {}

            with open(os.path.join(target_domain_path, fold, "facts.pl"), "r", encoding="utf-8") as f:
                lines = f.readlines()
                self._data["target"][fold]["facts"] = [line.strip() for line in lines]

            with open(os.path.join(target_domain_path, fold, "pos.pl"), "r", encoding="utf-8") as f:
                lines = f.readlines()

                if not self._target_predicate:
                    self._target_predicate = lines[0].split("(")[0]

                self._data["target"][fold]["pos"] = [line.strip() for line in lines]

            with open(os.path.join(target_domain_path, fold, "neg.pl"), "r", encoding="utf-8") as f:
                lines = f.readlines()
                self._data["target"][fold]["neg"] = [line.strip() for line in lines]

    def get_source_data(self, data_type):
        data = []
        for item in self._data["source"].values():
            for el in item.values():
                data += el[data_type]

        return data

    def get_target_data(self, data_type, folds):
        data = []
        for key, value in self._data["target"].items():
            if key in folds:
                data += value[data_type]

        return data

    def get_target_facts(self, fold):
        return self._data["target"][fold]["facts"]

    def generate_fold_splits(self, small_dataset=True):
        return generate_folds(self._target_domain_path, small_dataset)

    def get_target_domain_metadata(self):
        return self._target_domain_metadata

    def generate_train_srldatabase(self, train_folds: list[str], use_recursion: bool):
        db = Database()
        db.pos = self.get_source_data("pos") + self.get_target_data("pos", train_folds)
        db.neg = self.get_source_data("neg") + self.get_target_data("neg", train_folds)
        db.modes = self._modes
        db.facts = self.get_source_data("facts") + self.get_target_data("facts", train_folds)

        if use_recursion:
            db = db.generateRecursion()
        else:
            db = db.removeRecursion()

        return db

    def generate_test_srldatabase(self, test_folds, use_recursion):
        db = Database()
        db.pos = self.get_target_data("pos", test_folds)
        db.neg = self.get_target_data("neg", test_folds)
        db.modes = self._modes
        db.facts = self.get_target_data("facts", test_folds)

        if use_recursion:
            db = db.generateRecursion()
        else:
            db = db.removeRecursion()

        return db

    def generate_tl_database(self, train_folds, use_recursion):
        db = Database()
        db.pos = self.get_source_data("pos")
        db.neg = self.get_source_data("neg")
        db.modes = self._modes
        db.facts = self.get_source_data("facts")

        if use_recursion:
            source_database = db.generateRecursion()
        else:
            source_database = db.removeRecursion()

        db = Database()
        db.pos = self.get_target_data("pos", train_folds)
        db.neg = self.get_target_data("neg", train_folds)
        db.modes = self._modes
        db.facts = self.get_target_data("facts", train_folds)

        if use_recursion:
            target_database = db.generateRecursion()
        else:
            target_database = db.removeRecursion()

        weight_factory = WeightFactory()
        weight_strategy = weight_factory.getWeightStrategy("scalar", weight=1)

        database = Database.prepareTransferLearningDatabase(
            source_database,
            target_database,
            weight_strategy
        )

        return database

