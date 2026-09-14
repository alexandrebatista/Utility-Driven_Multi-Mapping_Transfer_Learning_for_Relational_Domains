import os
import json
from collections import defaultdict

from database.utils import graph_data_to_database, raw_data_to_database, convert_to_prolog_data, \
    convert_datatomap_to_prologdata
from database.database import DatabaseWrapper
from mapping.apply_mappings import read_source_domain_and_apply_mapping
from mapping.utils import select_mappings
from .model import RdnBoost
from .selector_model import SelectorModel

no_recursion_datasets = ["cora", "imdb", "uwcse"]

DATASETS_PATH = "./preprocessed/"


class TransferlearningManager:
    def __init__(self, base_path, target_domain_name):
        self._base_path = base_path

        self._target_domain_name = target_domain_name

        self._target_uses_recursion = not target_domain_name in no_recursion_datasets

        self._target_domain_path = os.path.join(DATASETS_PATH, target_domain_name)

        self._dataset = DatabaseWrapper()
        self._dataset.process_target(self._target_domain_path, target_domain_name)
        self._target_domain_metadata = self._dataset.get_target_domain_metadata()
        self._folds_split = self._dataset.generate_fold_splits(use_validation=True)

        self._mappings_path = os.path.join("./mappings", f"{target_domain_name}.json")

        self._source_domains = None
        self._selected_mappings = None
        self.gamma = 0.5

    def select_mappings(self, number_of_mappings, source_domain_name=None):
        with open(self._mappings_path, "r", encoding="utf-8") as f:
            mappings = json.load(f)

        selected_mappings = select_mappings(mappings, source_domain_name, number_of_mappings)
        self.reset(selected_mappings)

    def set_mappings(self, selected_mappings):
        self.reset(selected_mappings)

    def set_gamma(self, gamma):
        self.gamma = gamma

    def reset(self, selected_mappings):
        self._selected_mappings = selected_mappings
        self._source_domains = self._load_source_domains()
        self._dataset.reset_source_data()

    def _load_source_domains(self):
        mapped_data = {}

        for mapping_key, mapping in self._selected_mappings.items():
            source_domain_mapped_data = self._get_source_mapped_data(mapping)

            source_domain_converted_data = convert_datatomap_to_prologdata(source_domain_mapped_data)

            mapped_data[mapping_key] = source_domain_converted_data

        return mapped_data

    def _get_source_mapped_data(self, mapping):
        source_domain = mapping["source"]
        source_domain_data_path = os.path.join(DATASETS_PATH, source_domain)

        return read_source_domain_and_apply_mapping(
            source_domain_data_path,
            mapping,
            self._dataset.get_target_domain_metadata()
        )

    def _get_transfer_dataset(self, fold_configuration, sample_percentage=0.5, round_idx=0):
        """
        1. Calcula pesos GFL (qualidade individual).
        2. Aplica Redução Global (mantém apenas os melhores % do pool).
        3. Normaliza o Volume (Alpha) apenas sobre os sobreviventes.
        """
        from collections import defaultdict
        import os
        import math
        import json

        transfer_dataset = defaultdict(dict)
        fold_id = f"tr_{'-'.join(fold_configuration['train'])}_va_{'-'.join(fold_configuration['validation'])}"

        # ---------- 1. PREPARAÇÃO ----------
        train_target_data = {
            **self._dataset.get_target_facts(fold_configuration["train"]),
            **self._dataset.get_target_pos(fold_configuration["train"]),
            **self._dataset.get_target_neg(fold_configuration["train"]),
        }
        target_pos_count = len(train_target_data["pos"])
        target_neg_count = len(train_target_data["neg"])

        validation_target_data = {
            **self._dataset.get_target_facts(fold_configuration["validation"]),
            **self._dataset.get_target_pos(fold_configuration["validation"]),
            **self._dataset.get_target_neg(fold_configuration["validation"]),
        }

        # ---------- 2. ORÁCULO ----------
        scorer_model = SelectorModel(
            valid_predicates=self._target_domain_metadata["predicates"],
            target=self._target_domain_name,
            target_domain_metadata=self._target_domain_metadata,
            prefix_path=self._base_path,
            name=f"scorer_{self._target_domain_name}_{fold_id}",
        )
        scorer_model.train(train_target_data)
        scorer_model.test(validation_target_data)
        oracle_metrics = scorer_model.get_test_metrics()
        oracle_pr = max(0.0, min(1.0, oracle_metrics.get("auc_pr", 0.5)))

        confidence_factor = oracle_pr
        max_impact_ratio = 1.0 - (0.5 * confidence_factor)

        pool_source_pos = {}
        pool_source_neg = {}

        sampling_log = {
            "round": round_idx,
            "target_samples": {"pos": target_pos_count, "neg": target_neg_count},
            "parameters": {"sample_percentage": sample_percentage},
            "sources_info": {}
        }

        # ---------- 3. PESAGEM INDIVIDUAL (GFL) ----------
        num_mappings = len(self._source_domains)

        for mapping_key, mapped_data in self._source_domains.items():
            source_domain_name = self._selected_mappings[mapping_key]["source"]
            if not mapped_data["pos"] or not mapped_data["neg"]: continue

            scorer_model.test(mapped_data)
            try:
                estimates = scorer_model.load_estimates()
            except:
                continue

            gamma = self.gamma
            gfl_lambda = oracle_pr

            pos_set = set(mapped_data["pos"])

            sampling_log["sources_info"][mapping_key] = {
                "source": source_domain_name,
                "pos_before": len(mapped_data["pos"]),
                "neg_before": len(mapped_data["neg"])
            }

            for est in estimates:
                inst, c_i = est["raw"], max(min(est["score"], 1.0 - 1e-6), 1e-6)

                # Núcleo GFL
                gfl_core = -math.log(c_i) if gfl_lambda < 1e-4 else (1.0 - (c_i ** gfl_lambda)) / gfl_lambda
                w_gfl = max((confidence_factor * ((1.0 - c_i) ** gamma) * gfl_core) + ((1.0 - confidence_factor) * 1.0),
                            0.01)

                entry = {
                    "weight": w_gfl,
                    "mapping_key": mapping_key,
                    "source_name": source_domain_name,
                    "facts": mapped_data["facts"]
                }

                if inst in pos_set:
                    pool_source_pos[inst] = entry
                else:
                    pool_source_neg[inst] = entry

        # ---------- 4. REDUÇÃO NO AGREGADO (RANKING GLOBAL) ----------
        selected_pos_ids = list(pool_source_pos.keys())
        selected_neg_ids = list(pool_source_neg.keys())

        if num_mappings > 1:
            k_pos = math.ceil(len(pool_source_pos) * sample_percentage)
            k_neg = math.ceil(len(pool_source_neg) * sample_percentage)

            if len(selected_pos_ids) > k_pos:
                selected_pos_ids = sorted(selected_pos_ids, key=lambda x: pool_source_pos[x]["weight"], reverse=True)[
                                   :k_pos]
            if len(selected_neg_ids) > k_neg:
                selected_neg_ids = sorted(selected_neg_ids, key=lambda x: pool_source_neg[x]["weight"], reverse=True)[
                                   :k_neg]

        sampling_log["global_reduction"] = {
            "total_pool_pos": len(pool_source_pos), "total_pool_neg": len(pool_source_neg),
            "selected_pos": len(selected_pos_ids), "selected_neg": len(selected_neg_ids)
        }

        # ---------- 5. NORMALIZAÇÃO DE VOLUME (ALPHA) ----------
        # Agora calculamos o somatório dos pesos apenas de quem não foi removido

        final_weights = {}  # inst -> final_weight

        # Estrutura auxiliar para somar pesos dos selecionados por mapping
        sum_weights_per_mapping = defaultdict(lambda: {"pos": 0.0, "neg": 0.0})

        for inst in selected_pos_ids:
            sum_weights_per_mapping[pool_source_pos[inst]["mapping_key"]]["pos"] += pool_source_pos[inst]["weight"]
        for inst in selected_neg_ids:
            sum_weights_per_mapping[pool_source_neg[inst]["mapping_key"]]["neg"] += pool_source_neg[inst]["weight"]

        max_allowed_pos = float(target_pos_count) * max_impact_ratio
        max_allowed_neg = float(target_neg_count) * max_impact_ratio

        # Aplicar Scaling Alpha e Distribuir no Dataset
        for inst in selected_pos_ids:
            meta = pool_source_pos[inst]
            m_key = meta["mapping_key"]
            s_name = meta["source_name"]

            # Fator de escala baseado apenas no que restou deste mapping
            sum_w = sum_weights_per_mapping[m_key]["pos"]
            alpha_scale = max_allowed_pos / sum_w if sum_w > 0 else 1.0

            w_final = max(meta["weight"] * alpha_scale, 0.001)
            final_weights[inst] = w_final

            if m_key not in transfer_dataset[s_name]:
                transfer_dataset[s_name][m_key] = {"facts": meta["facts"], "pos": [], "neg": []}
            transfer_dataset[s_name][m_key]["pos"].append(inst)

        for inst in selected_neg_ids:
            meta = pool_source_neg[inst]
            m_key = meta["mapping_key"]
            s_name = meta["source_name"]

            sum_w = sum_weights_per_mapping[m_key]["neg"]
            alpha_scale = max_allowed_neg / sum_w if sum_w > 0 else 1.0

            w_final = max(meta["weight"] * alpha_scale, 0.001)
            final_weights[inst] = w_final

            if m_key not in transfer_dataset[s_name]:
                transfer_dataset[s_name][m_key] = {"facts": meta["facts"], "pos": [], "neg": []}
            transfer_dataset[s_name][m_key]["neg"].append(inst)

        # ---------- 6. EXPORTAÇÃO ----------
        global_instance_weights = {inst: 1.0 for inst in train_target_data["pos"]}
        global_instance_weights.update({inst: 1.0 for inst in train_target_data["neg"]})
        global_instance_weights.update(final_weights)

        weights_dir = os.path.join(self._base_path, "weights")
        os.makedirs(weights_dir, exist_ok=True)

        with open(os.path.join(weights_dir, f"sampling_log_fold_{fold_id}_rd_{round_idx}.json"), "w") as f:
            json.dump(sampling_log, f, indent=4)

        single_weights_file_path = os.path.join(weights_dir, f"weights_global_{self._target_domain_name}_{fold_id}.txt")
        with open(single_weights_file_path, "w") as f:
            for inst, w in global_instance_weights.items():
                f.write(f"{inst}\t{w:.6f}\n")

        scorer_model.delete()
        return transfer_dataset, single_weights_file_path

    def _gfl_training_workflow(self, fold_configuration, weights_file_path):
        transfer_learning_train_dataset = self._dataset.generate_train_srldatabase(
            fold_configuration["train"],
            use_recursion=self._target_uses_recursion,
        )

        transfer_learning_test_dataset = self._dataset.generate_test_srldatabase(
            fold_configuration["test"],
            use_recursion=self._target_uses_recursion,
        )

        model = RdnBoost(
            target_domain_metadata=self._target_domain_metadata,
            prefix_path=self._base_path,
            name="gfl_model",
            weights_file_path=weights_file_path
        )

        model.train(transfer_learning_train_dataset)
        train_metrics = model.get_train_metrics()

        model.test(transfer_learning_test_dataset, convert=False)
        test_metrics = model.get_test_metrics()

        model.delete()

        return train_metrics, test_metrics

    def default_transfer_learning(self, rounds: int = 3):
        out = []

        data = defaultdict(dict)
        for mapping_key, mapped_data in self._source_domains.items():
            source_domain_name = self._selected_mappings[mapping_key]["source"]
            data[source_domain_name].update({mapping_key: mapped_data})

        self._dataset.process_source_from_data(data)

        for fold_configuration in self._folds_split:
            train_database = self._dataset.generate_train_srldatabase(
                fold_configuration["train"],
                use_recursion=self._target_uses_recursion,
            )

            test_database = self._dataset.generate_test_srldatabase(
                fold_configuration["test"],
                use_recursion=self._target_uses_recursion
            )

            for _ in range(rounds):
                model = RdnBoost(
                    target_domain_metadata=self._target_domain_metadata,
                    prefix_path=self._base_path,
                    name="default_model"
                )

                model.train(train_database)
                train_metrics = model.get_train_metrics()

                model.test(test_database, convert=False)
                test_metrics = model.get_test_metrics()

                model.delete()

                out.append(
                    {
                        "fold_configuration": fold_configuration,
                        "train": train_metrics,
                        "test": test_metrics
                    }
                )

        self._dataset.reset_source_data()

        return out

    def gfl_transfer_learning(self, rounds: int = 3):
        out = []
        tl_not_possible = []

        for fold_configuration in self._folds_split:
            for round_index in range(rounds):
                self._dataset.reset_source_data()

                result = {"transfer_learning": False, "fold_configuration": fold_configuration, "log": ""}
                # Generate dataset with misclassified by the model

                if fold_configuration["train"][0] not in tl_not_possible:
                    # Get transfer dataset
                    transfer_learning_raw_dataset, weights_file_path = self._get_transfer_dataset(
                        fold_configuration=fold_configuration,
                        round_idx=round_index
                    )

                    if transfer_learning_raw_dataset is not None:
                        result["transfer_learning"] = True
                        self._dataset.process_source_from_data(transfer_learning_raw_dataset)

                    else:
                        tl_not_possible.append(fold_configuration["train"][0])
                        result["log"] += "Transfer-learning dataset could not be generated."

                tl_train_metrics, tl_test_metrics = self._gfl_training_workflow(
                    fold_configuration=fold_configuration,
                    weights_file_path=weights_file_path
                )

                os.remove(weights_file_path)

                self._dataset.reset_source_data()

                result["train"] = tl_train_metrics
                result["test"] = tl_test_metrics

                out.append(result)

        return out

    def _save_results(self, results):
        """Salva os resultados do transfer learning em um arquivo JSON."""
        import shutil

        results_dir = os.path.join(self._base_path, "results")
        os.makedirs(results_dir, exist_ok=True)

        source_domains = "_".join(results["source"])
        mapping_ids = "-".join(self._selected_mappings.keys())
        base_filename = f"{source_domains}_{self._target_domain_name}_{mapping_ids}.json"
        filepath = os.path.join(results_dir, base_filename)

        # Se arquivo já existe, adiciona número no final
        if os.path.exists(filepath):
            counter = 1
            name, ext = os.path.splitext(base_filename)
            while os.path.exists(os.path.join(results_dir, f"{name}_{counter}{ext}")):
                counter += 1
            filepath = os.path.join(results_dir, f"{name}_{counter}{ext}")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Resultados salvos em: {filepath}")
        return filepath

    def perform_transfer_learning(self):
        source_domains = [mapping["source"] for mapping in self._selected_mappings.values()]

        results = {
            "target": self._target_domain_name,
            "source": source_domains,
            "mappings": list(self._selected_mappings.values())
        }

        gfl_tl_out = self.gfl_transfer_learning()
        default_tl_out = self.default_transfer_learning()

        results["default_transfer_learning_results"] = default_tl_out
        results["gfl_transfer_learning_results"] = gfl_tl_out

        # Salva os resultados em arquivo JSON
        self._save_results(results)

        return results