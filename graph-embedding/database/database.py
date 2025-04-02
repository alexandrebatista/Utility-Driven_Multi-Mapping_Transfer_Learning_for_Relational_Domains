import os
from srlearn import Database
from srlearn.weight import WeightFactory


class DatabaseWrapper:
    def __init__(self):
        self._modes = []

        self._target_predicate = None

        self._data = {
            "source": {

            },
            "target": {

            }
        }

    def process_source(self, path):
        for source_domain in os.listdir(path):
            self._data["source"][source_domain] = {}

            for mapping_folder in os.listdir(os.path.join(path, source_domain)):
                # is data
                if os.path.isdir(os.path.join(path, source_domain, mapping_folder)):
                    self._data["source"][source_domain][mapping_folder] = {"facts.pl": [], "pos.pl": [], "neg.pl": []}

                    for file in os.listdir(os.path.join(path, source_domain, mapping_folder)):
                        with open(os.path.join(path, source_domain, mapping_folder, file), "r", encoding="utf-8") as f:
                            lines = f.readlines()
                            self._data["source"][source_domain][mapping_folder][file].extend([line.strip() for line in lines])

                # is file (mapping definition)


    def process_target(self, target_domain_path):
        for fold in os.listdir(target_domain_path):
            if not self._modes:
                with open(os.path.join(target_domain_path, fold, "modes.pl"), "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    self._modes.extend([line.strip() for line in lines])

            self._data["target"][fold] = {}

            with open(os.path.join(target_domain_path, fold, "facts.pl"), "r", encoding="utf-8") as f:
                lines = f.readlines()
                self._data["target"][fold]["facts.pl"] = [line.strip() for line in lines]

            with open(os.path.join(target_domain_path, fold, "pos.pl"), "r", encoding="utf-8") as f:
                lines = f.readlines()

                if not self._target_predicate:
                    self._target_predicate = lines[0].split("(")[0]

                self._data["target"][fold]["pos.pl"] = [line.strip() for line in lines]

            with open(os.path.join(target_domain_path, fold, "neg.pl"), "r", encoding="utf-8") as f:
                lines = f.readlines()
                self._data["target"][fold]["neg.pl"] = [line.strip() for line in lines]

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

    def generate_train_srldatabase(self, train_folds, use_recursion):
        db = Database()
        db.pos = self.get_source_data("pos.pl") + self.get_target_data("pos.pl", train_folds)
        db.neg = self.get_source_data("neg.pl") + self.get_target_data("neg.pl", train_folds)
        db.modes = self._modes
        db.facts = self.get_source_data("facts.pl") + self.get_target_data("facts.pl", train_folds)

        if use_recursion:
            db = db.generateRecursion()
        else:
            db = db.removeRecursion()

        return db

    def generate_test_srldatabase(self, test_folds, use_recursion):
        db = Database()
        db.pos = self.get_target_data("pos.pl", test_folds)
        db.neg = self.get_target_data("neg.pl", test_folds)
        db.modes = self._modes
        db.facts = self.get_target_data("facts.pl", test_folds)

        if use_recursion:
            db = db.generateRecursion()
        else:
            db = db.removeRecursion()

        return db

    def generate_tl_database(self, train_folds, use_recursion):
        db = Database()
        db.pos = self.get_source_data("pos.pl")
        db.neg = self.get_source_data("neg.pl")
        db.modes = self._modes
        db.facts = self.get_source_data("facts.pl")

        if use_recursion:
            source_database = db.generateRecursion()
        else:
            source_database = db.removeRecursion()

        db = Database()
        db.pos = self.get_target_data("pos.pl", train_folds)
        db.neg = self.get_target_data("neg.pl", train_folds)
        db.modes = self._modes
        db.facts = self.get_target_data("facts.pl", train_folds)

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

