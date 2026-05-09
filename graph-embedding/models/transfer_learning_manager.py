import os
import json
from collections import defaultdict

from database.utils import graph_data_to_database, raw_data_to_database, convert_to_prolog_data, convert_datatomap_to_prologdata
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

    def select_mappings(self, number_of_mappings, source_domain_name=None):
        with open(self._mappings_path, "r", encoding="utf-8") as f:
            mappings = json.load(f)

        selected_mappings = select_mappings(mappings, source_domain_name, number_of_mappings)
        self.reset(selected_mappings)

    def set_mappings(self, selected_mappings):
        self.reset(selected_mappings)

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

    def _get_transfer_dataset(self, fold_configuration, sample_multiplier=2.0, round_idx=0):
        """
        Versão Agregada: Transferência Epistêmica usando GFL.
        - Calcula pesos para todos os sources.
        - Aplica redução (Top-K) sobre o POOL GLOBAL de amostras source.
        - Gera logs detalhados de amostragem por rodada.
        """
        from collections import defaultdict
        import os
        import math
        import json

        transfer_dataset = defaultdict(dict)
        fold_id = f"tr_{'-'.join(fold_configuration['train'])}_va_{'-'.join(fold_configuration['validation'])}"

        # ---------- 1. PREPARAÇÃO DOS DADOS ----------
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

        # ---------- 2. ORÁCULO E DIAGNÓSTICO ----------
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

        # Estruturas para o Pool Global
        pool_source_pos = {} # inst -> {weight: w, mapping_key: k, source_name: s, facts: f}
        pool_source_neg = {}
        
        # Estatísticas para o LOG
        sampling_log = {
            "round": round_idx,
            "target_samples": {"pos": target_pos_count, "neg": target_neg_count},
            "sources_before_reduction": {},
            "global_reduction": {}
        }

        # ---------- 3. COLETA E PESAGEM DE TODOS OS SOURCES ----------
        num_mappings = len(self._source_domains)

        for mapping_key, mapped_data in self._source_domains.items():
            source_domain_name = self._selected_mappings[mapping_key]["source"]
            if not mapped_data["pos"] or not mapped_data["neg"]: continue

            scorer_model.test(mapped_data)
            try:
                estimates = scorer_model.load_estimates()
            except: continue

            # GFL Params
            gamma, gfl_lambda = 0.5, oracle_pr
            pos_set = set(mapped_data["pos"])
            
            temp_weights_pos, temp_weights_neg = {}, {}
            sum_w_pos, sum_w_neg = 0.0, 0.0

            for est in estimates:
                inst, c_i = est["raw"], max(min(est["score"], 1.0 - 1e-6), 1e-6)
                gfl_core = -math.log(c_i) if gfl_lambda < 1e-4 else (1.0 - (c_i ** gfl_lambda)) / gfl_lambda
                w = max((confidence_factor * ((1.0 - c_i) ** gamma) * gfl_core) + ((1.0 - confidence_factor) * 1.0), 0.01)

                if inst in pos_set:
                    temp_weights_pos[inst] = w
                    sum_w_pos += w
                else:
                    temp_weights_neg[inst] = w
                    sum_w_neg += w

            # Normalização Estratificada Individual (Alpha)
            s_pos = (target_pos_count * max_impact_ratio) / sum_w_pos if sum_w_pos > 0 else 1.0
            s_neg = (target_neg_count * max_impact_ratio) / sum_w_neg if sum_w_neg > 0 else 1.0

            sampling_log["sources_before_reduction"][mapping_key] = {
                "source": source_domain_name,
                "pos": len(temp_weights_pos), "neg": len(temp_weights_neg)
            }

            # Aloca no Pool Global
            for inst, w in temp_weights_pos.items():
                pool_source_pos[inst] = {"weight": max(w * s_pos, 0.001), "mapping_key": mapping_key, "source_name": source_domain_name, "facts": mapped_data["facts"]}
            for inst, w in temp_weights_neg.items():
                pool_source_neg[inst] = {"weight": max(w * s_neg, 0.001), "mapping_key": mapping_key, "source_name": source_domain_name, "facts": mapped_data["facts"]}

        # ---------- 4. REDUÇÃO NO AGREGADO (GLOBAL RANKING) ----------
        final_selected_pos = list(pool_source_pos.keys())
        final_selected_neg = list(pool_source_neg.keys())

        if num_mappings > 1:
            limit_pos = math.ceil(target_pos_count * sample_multiplier)
            limit_neg = math.ceil(target_neg_count * sample_multiplier)

            if len(final_selected_pos) > limit_pos:
                final_selected_pos = sorted(final_selected_pos, key=lambda x: pool_source_pos[x]["weight"], reverse=True)[:limit_pos]
            if len(final_selected_neg) > limit_neg:
                final_selected_neg = sorted(final_selected_neg, key=lambda x: pool_source_neg[x]["weight"], reverse=True)[:limit_neg]

        # Registro do Log Global
        sampling_log["global_reduction"] = {
            "total_source_pos_before": len(pool_source_pos),
            "total_source_neg_before": len(pool_source_neg),
            "total_source_pos_after": len(final_selected_pos),
            "total_source_neg_after": len(final_selected_neg),
            "removed_pos": len(pool_source_pos) - len(final_selected_pos),
            "removed_neg": len(pool_source_neg) - len(final_selected_neg)
        }

        # ---------- 5. DISTRIBUIÇÃO E PESOS FINAIS ----------
        global_instance_weights = {inst: 1.0 for inst in train_target_data["pos"]}
        global_instance_weights.update({inst: 1.0 for inst in train_target_data["neg"]})

        # Reconstrói o transfer_dataset apenas com os sobreviventes
        set_selected_pos = set(final_selected_pos)
        set_selected_neg = set(final_selected_neg)

        for inst in final_selected_pos:
            meta = pool_source_pos[inst]
            m_key, s_name = meta["mapping_key"], meta["source_name"]
            if m_key not in transfer_dataset[s_name]:
                transfer_dataset[s_name][m_key] = {"facts": meta["facts"], "pos": [], "neg": []}
            transfer_dataset[s_name][m_key]["pos"].append(inst)
            global_instance_weights[inst] = meta["weight"]

        for inst in final_selected_neg:
            meta = pool_source_neg[inst]
            m_key, s_name = meta["mapping_key"], meta["source_name"]
            if m_key not in transfer_dataset[s_name]:
                transfer_dataset[s_name][m_key] = {"facts": meta["facts"], "pos": [], "neg": []}
            transfer_dataset[s_name][m_key]["neg"].append(inst)
            global_instance_weights[inst] = meta["weight"]

        # ---------- 6. EXPORTAÇÃO ----------
        weights_dir = os.path.join(self._base_path, "weights")
        os.makedirs(weights_dir, exist_ok=True)
        
        # Salva Log de Amostragem
        log_path = os.path.join(weights_dir, f"sampling_log_fold_{fold_id}_rd_{round_idx}.json")
        with open(log_path, "w") as f:
            json.dump(sampling_log, f, indent=4)

        # Salva Pesos
        single_weights_file_path = os.path.join(weights_dir, f"weights_global_{self._target_domain_name}_{fold_id}.txt")
        with open(single_weights_file_path, "w") as f:
            for inst, w in global_instance_weights.items():
                f.write(f"{inst}\t{w:.6f}\n")

        print(f"   -> [Agregado] Redução Global Concluída. Log: {log_path}")
        scorer_model.delete()
        return transfer_dataset, single_weights_file_path

    def _kg_training_workflow(self, fold_configuration, weights_file_path):
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
            name="kg_model",
            weights_file_path=weights_file_path
        )

        model.train(transfer_learning_train_dataset)
        train_metrics = model.get_train_metrics()

        model.test(transfer_learning_test_dataset, convert=False)
        test_metrics = model.get_test_metrics()

        model.delete()

        return train_metrics, test_metrics

    def default_transfer_learning(self, rounds:int = 3):
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

    def kg_transfer_learning(self, rounds: int = 3):
        out = []
        tl_not_possible = []

        for fold_configuration in self._folds_split:
            for _ in range(rounds):
                self._dataset.reset_source_data()

                result = {"transfer_learning": False, "fold_configuration": fold_configuration, "log": ""}
                # Generate dataset with misclassified by the model

                if fold_configuration["train"][0] not in tl_not_possible:
                    # Get transfer dataset
                    transfer_learning_raw_dataset, weights_file_path = self._get_transfer_dataset(
                        fold_configuration=fold_configuration
                    )

                    if transfer_learning_raw_dataset is not None:
                        result["transfer_learning"] = True
                        self._dataset.process_source_from_data(transfer_learning_raw_dataset)

                    else:
                        tl_not_possible.append(fold_configuration["train"][0])
                        result["log"] += "Transfer-learning dataset could not be generated."

                tl_train_metrics, tl_test_metrics = self._kg_training_workflow(
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

        kg_tl_out = self.kg_transfer_learning()
        default_tl_out = self.default_transfer_learning()

        results["default_transfer_learning_results"] = default_tl_out
        results["kg_transfer_learning_results"] = kg_tl_out
        
        # Salva os resultados em arquivo JSON
        self._save_results(results)
        
        return results