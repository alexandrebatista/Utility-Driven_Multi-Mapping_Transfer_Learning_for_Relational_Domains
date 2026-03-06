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

        self._best_models_found_path = os.path.join(self._base_path, f"best_models_found.json")
        self._best_models_found = self._load_best_models_found()

    def _load_best_models_found(self):
        best_models_found = {}

        folds = sorted(
            [d for d in os.listdir(os.path.join(DATASETS_PATH, self._target_domain_name)) if d.startswith("fold")]
        )

        if os.path.exists(self._best_models_found_path):
            with open(self._best_models_found_path, 'r') as f:
                try:
                    best_models_found = json.load(f)
                except json.JSONDecodeError:
                    pass

            if self._target_domain_name in best_models_found:
                return best_models_found

        best_models_found.update({
            self._target_domain_name: {fold: 0 for fold in folds}
        })

        with open(self._best_models_found_path, 'w') as f:
            json.dump(best_models_found, f, indent=4)

        return best_models_found

    def _update_best_models_found(self, fold, best_model_was_found):
        self._best_models_found[self._target_domain_name][fold] = 1 if best_model_was_found else -1

        with open(self._best_models_found_path, 'w') as f:
            json.dump(self._best_models_found, f, indent=4)

    def _best_model_was_not_found(self, fold):
        return self._best_models_found[self._target_domain_name][fold] == -1

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

            mapped_data = {mapping_key: source_domain_converted_data}

        return mapped_data

    def _get_source_mapped_data(self, mapping):
        source_domain = mapping["source"]
        source_domain_data_path = os.path.join(DATASETS_PATH, source_domain)

        return read_source_domain_and_apply_mapping(
            source_domain_data_path,
            mapping,
            self._dataset.get_target_domain_metadata()
        )

    def _get_transfer_dataset(self, fold_configuration):
        """
        Versão Definitiva: Soft-Attention Relacional com Modulador Epistêmico e Arquivo Único.
        - Preserva 100% da topologia (sem pruning ou oversampling relacional).
        - Pesos baseados em Consistência Semântica (Target vs. Gabarito do Source).
        - Bypass automático em caso de colapso do Target (Cold-Start).
        - Normalização de volume estratificada (proteção de classes minoritárias).
        - Exporta um ÚNICO arquivo de pesos consolidado para o Java.
        """
        from collections import defaultdict
        import os

        transfer_dataset = defaultdict(dict)

        # ---------- 1. PREPARAÇÃO DOS DADOS ----------
        train_target_data = {
            **self._dataset.get_target_facts(fold_configuration["train"]),
            **self._dataset.get_target_pos(fold_configuration["train"]),
            **self._dataset.get_target_neg(fold_configuration["train"]),
        }

        validation_target_data = {
            **self._dataset.get_target_facts(fold_configuration["validation"]),
            **self._dataset.get_target_pos(fold_configuration["validation"]),
            **self._dataset.get_target_neg(fold_configuration["validation"]),
        }

        fold_id = f"tr_{'-'.join(fold_configuration['train'])}_va_{'-'.join(fold_configuration['validation'])}"

        # ---------- 2. ORÁCULO E DIAGNÓSTICO DO TARGET ----------
        print(f"\n[Soft-Attention] Diagnosticando o domínio Alvo (Fold: {fold_id})...")
        scorer_model = SelectorModel(
            valid_predicates=self._target_domain_metadata["predicates"],
            target=self._target_domain_name,
            target_domain_metadata=self._target_domain_metadata,
            prefix_path=self._base_path,
            name=f"scorer_{self._target_domain_name}_{fold_id}",
        )
        scorer_model.train(train_target_data)

        # Avalia a competência do Target para calibrar o rigor do filtro
        scorer_model.test(validation_target_data)
        oracle_metrics = scorer_model.get_test_metrics()
        oracle_pr = oracle_metrics.get("auc_pr", 0.5)

        # Definição do Modulador Epistêmico (Teto de Impacto e Bypass)
        if oracle_pr >= 0.75:
            oracle_status = "ESPECIALISTA"
            max_impact_ratio = 0.5  # Source atua apenas como assistente (Max 50%)
        elif oracle_pr >= 0.60:
            oracle_status = "MEDIANO"
            max_impact_ratio = 1.0  # Paridade de volume (1:1), com filtro de consistência
        else:
            oracle_status = "COLAPSO_TARGET (Bypass Ativado)"
            max_impact_ratio = 1.0  # Paridade de volume (1:1), SEM filtro (aceita tudo do Source com peso 1.0)

        print(f"   -> Status: {oracle_status} | AUC-PR: {oracle_pr:.4f} | Teto de Impacto: {max_impact_ratio}x")

        # =====================================================================
        # DICIONÁRIO GLOBAL DE PESOS (Para o arquivo único)
        # =====================================================================
        global_instance_weights = {}

        # 3.1. Âncora do Target (Peso Absoluto 1.0 para os dados nativos)
        for pos_inst in train_target_data["pos"]:
            global_instance_weights[pos_inst] = 1.0
        for neg_inst in train_target_data["neg"]:
            global_instance_weights[neg_inst] = 1.0

        # ---------- 3. AVALIAÇÃO DAS ORIGENS (SOURCES) ----------
        for mapping_key, mapped_data in self._source_domains.items():
            source_domain_name = self._selected_mappings[mapping_key]["source"]

            if not mapped_data["pos"] or not mapped_data["neg"]:
                continue

            source_raw_weights_pos = {}
            source_raw_weights_neg = {}
            raw_source_weight_sum_pos = 0.0
            raw_source_weight_sum_neg = 0.0
            pos_source_set = set(mapped_data["pos"])

            # 3.2. Bypass ou Avaliação do Oráculo
            if oracle_status == "COLAPSO_TARGET (Bypass Ativado)":
                print(f"   [{source_domain_name}] Bypass: Transferindo Source com peso total 1.0.")
                for raw_instance in mapped_data["pos"]:
                    source_raw_weights_pos[raw_instance] = 1.0
                    raw_source_weight_sum_pos += 1.0
                for raw_instance in mapped_data["neg"]:
                    source_raw_weights_neg[raw_instance] = 1.0
                    raw_source_weight_sum_neg += 1.0
            else:
                scorer_model.test(mapped_data)
                try:
                    estimates = scorer_model.load_estimates()
                except Exception as e:
                    print(f"   [ERRO] Falha ao carregar estimates: {e}")
                    continue

                for est in estimates:
                    raw_instance = est["raw"]
                    consistency = est["score"]

                    if oracle_status == "ESPECIALISTA":
                        w = consistency if consistency >= 0.7 else 0.01
                    else:  # MEDIANO
                        w = consistency if consistency >= 0.6 else (0.05 if consistency <= 0.4 else 0.5)

                    w = max(w, 0.01)

                    if raw_instance in pos_source_set:
                        source_raw_weights_pos[raw_instance] = w
                        raw_source_weight_sum_pos += w
                    else:
                        source_raw_weights_neg[raw_instance] = w
                        raw_source_weight_sum_neg += w

            # 3.3. Normalização Estratificada de Volume
            target_pos_weight = float(len(train_target_data["pos"]))
            target_neg_weight = float(len(train_target_data["neg"]))

            max_allowed_source_pos = target_pos_weight * max_impact_ratio
            max_allowed_source_neg = target_neg_weight * max_impact_ratio

            scaling_factor_pos = 1.0
            if raw_source_weight_sum_pos > max_allowed_source_pos and raw_source_weight_sum_pos > 0:
                scaling_factor_pos = max_allowed_source_pos / raw_source_weight_sum_pos

            scaling_factor_neg = 1.0
            if raw_source_weight_sum_neg > max_allowed_source_neg and raw_source_weight_sum_neg > 0:
                scaling_factor_neg = max_allowed_source_neg / raw_source_weight_sum_neg

            # Salva no dicionário GLOBAL
            for inst, w in source_raw_weights_pos.items():
                global_instance_weights[inst] = max(w * scaling_factor_pos, 0.001)

            for inst, w in source_raw_weights_neg.items():
                global_instance_weights[inst] = max(w * scaling_factor_neg, 0.001)

            print(f"   [{source_domain_name}] Scaling POS: {scaling_factor_pos:.4f} | NEG: {scaling_factor_neg:.4f}")

            # Salva os dados no dataset de transferência
            transfer_dataset[source_domain_name][mapping_key] = {
                "facts": mapped_data["facts"],
                "pos": mapped_data["pos"],
                "neg": mapped_data["neg"]
            }

        # ---------- 4. EXPORTAÇÃO DO ARQUIVO ÚNICO DE PESOS ----------
        weights_dir = os.path.join(self._base_path, "weights")
        os.makedirs(weights_dir, exist_ok=True)

        # O nome do arquivo agora reflete apenas o Target e o Fold (já que unifica as origens)
        single_weights_file_path = os.path.join(weights_dir, f"weights_global_{self._target_domain_name}_{fold_id}.txt")

        with open(single_weights_file_path, "w") as f:
            for inst, w in global_instance_weights.items():
                f.write(f"{inst}\t{w:.6f}\n")

        print(f"\n[Soft-Attention] Arquivo ÚNICO de pesos gerado: {single_weights_file_path}")

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