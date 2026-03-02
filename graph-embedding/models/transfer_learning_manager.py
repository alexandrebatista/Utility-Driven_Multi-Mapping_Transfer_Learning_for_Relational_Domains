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
        Versão Definitiva para a Dissertação: Oversampling + Limiar Dinâmico + Trava de Treino.
        - Mantém a topologia 100% intacta (nunca corta nós do Source).
        - Duplica a Elite (Oversampling) para atuar como peso estrutural.
        - Usa Validação Empírica com Limiar Dinâmico de Risco.
        - TRAVA DE SANIDADE: Rejeita candidatos que causem Catastrophic Forgetting no Treino.
        """
        from collections import defaultdict

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

        # Identificador único do fold (para isolamento de artefatos em disco)
        fold_id = f"tr_{'-'.join(fold_configuration['train'])}_va_{'-'.join(fold_configuration['validation'])}"

        # ---------- 2. AVALIAÇÃO DO BASELINE (TARGET-ONLY) ----------
        baseline_model = SelectorModel(
            valid_predicates=self._target_domain_metadata["predicates"],
            target=self._target_domain_name,
            target_domain_metadata=self._target_domain_metadata,
            prefix_path=self._base_path,
            name=f"baseline_{self._target_domain_name}_{fold_id}",
        )
        baseline_model.train(train_target_data)

        # Coleta as métricas de TREINO e VALIDAÇÃO do Baseline
        no_transfer_train = baseline_model.get_train_metrics()
        baseline_model.test(validation_target_data)
        no_transfer_val = baseline_model.get_test_metrics()
        baseline_model.delete()

        # ---------- 3. FUNÇÃO DE DECISÃO ROBUSTA ----------
        def better(cand_val, cand_train, best_val, best_train):
            # 3.1. TRAVA DE SANIDADE (Anti-Catastrophic Forgetting)
            # Se a nova escolha destrói a performance no grafo principal (treino)
            # em mais de 3%, ela é uma fraude (overfitting na validação). Rejeita.
            if cand_train.get("auc_pr", 0) < (best_train.get("auc_pr", 0) - 0.03):
                return False

            # 3.2. LIMIAR DINÂMICO DE RISCO NA VALIDAÇÃO
            baseline_pr = best_val.get("auc_pr", 0)
            baseline_roc = best_val.get("auc_roc", 0)

            if baseline_pr >= 0.80:
                min_improvement = 0.05  # Risco alto: exige 5% de ganho real
            elif baseline_pr >= 0.65:
                min_improvement = 0.02  # Risco médio: exige 2% de ganho real
            else:
                min_improvement = 0.01  # Desespero: aceita 1% de ajuda

            pr_melhorou_muito = cand_val.get("auc_pr", 0) >= (baseline_pr + min_improvement)
            empate_tecnico_pr = cand_val.get("auc_pr", 0) >= baseline_pr
            roc_melhorou_muito = cand_val.get("auc_roc", 0) >= (baseline_roc + min_improvement)

            return pr_melhorou_muito or (empate_tecnico_pr and roc_melhorou_muito)

        # ---------- 4. AVALIAÇÃO DAS ORIGENS (SOURCES) ----------
        for mapping_key, mapped_data in self._source_domains.items():
            source_domain_name = self._selected_mappings[mapping_key]["source"]
            valid_predicates = [
                p for p in self._selected_mappings[mapping_key]["predicate_mappings"].values() if p is not None
            ]

            avail_pos = len(mapped_data["pos"])
            avail_neg = len(mapped_data["neg"])
            if avail_pos == 0 or avail_neg == 0:
                print(f"[{source_domain_name}] Skip (sem pos/neg no source).")
                continue

            # Inicia o rastreador de melhores escolhas cravado no Baseline Target-only
            best_val = {
                "auc_pr": no_transfer_val.get("auc_pr", float("-inf")),
                "auc_roc": no_transfer_val.get("auc_roc", float("-inf")),
                "choice": "Target-only",
                "pos": [],
                "neg": [],
            }
            best_train = no_transfer_train.copy()

            # ---------- 4.1. MODELO "FULL SOURCE" ----------
            selector_model = SelectorModel(
                valid_predicates=valid_predicates,
                target=self._target_domain_name,
                target_domain_metadata=self._target_domain_metadata,
                prefix_path=self._base_path,
                name=f"selector_full_{mapping_key}_{fold_id}",
            )

            unified_data = {
                "facts": train_target_data["facts"] + mapped_data["facts"],
                "pos": train_target_data["pos"] + mapped_data["pos"],
                "neg": train_target_data["neg"] + mapped_data["neg"],
            }

            selector_model.train(unified_data)
            full_train_metrics = selector_model.get_train_metrics()

            selector_model.test(validation_target_data)
            full_val_metrics = selector_model.get_test_metrics()

            # Avalia a Transferência Total usando a Trava e o Limiar Dinâmico
            if better(full_val_metrics, full_train_metrics, best_val, best_train):
                best_val.update({
                    "auc_pr": full_val_metrics["auc_pr"],
                    "auc_roc": full_val_metrics["auc_roc"],
                    "choice": "Full Source",
                    "pos": mapped_data["pos"],
                    "neg": mapped_data["neg"],
                })
                best_train = full_train_metrics.copy()

            # ---------- 4.2. GERAÇÃO DE ESTIMATIVAS DA ELITE ----------
            selector_model.test(mapped_data)

            # Sanity check ajustado (Aviso em vez de Erro Fatal para proteger contra nós isolados)
            try:
                est = selector_model.load_estimates()
                expected = avail_pos + avail_neg
                if len(est) != expected:
                    print(
                        f"   [AVISO] {source_domain_name}: Recebeu {len(est)} estimates, esperado {expected}. O BoostSRL ignorou nós isolados.")
            except Exception as e:
                print(f"   [ERRO] Falha ao carregar estimates para {source_domain_name}: {e}")
                selector_model.delete()
                continue  # Pula para a próxima origem em vez de matar a execução

            # ---------- 4.3. REFORÇO ESTRUTURAL (OVERSAMPLING) ----------
            # Testamos duplicar os top 25% e os top 50% mais confiáveis.
            oversample_grid = [0.25, 0.50]
            mix_extreme = 1.0  # Queremos a elite pura (top scores)

            for frac in oversample_grid:
                n_pos_elite = max(1, int(avail_pos * frac))
                n_neg_elite = max(2, int(avail_neg * frac))

                cand_pos_elite, cand_neg_elite = selector_model.get_best_samples(
                    n_pos=n_pos_elite, n_neg=n_neg_elite, mix_extreme=mix_extreme
                )

                elite_pos_raw = [s["raw"] for s in cand_pos_elite]
                elite_neg_raw = [s["raw"] for s in cand_neg_elite]

                # O Source Original é MANTIDO. A Elite é SOMADA (Duplicada).
                reinforced_pos = mapped_data["pos"] + elite_pos_raw
                reinforced_neg = mapped_data["neg"] + elite_neg_raw

                tester_model = SelectorModel(
                    valid_predicates=valid_predicates,
                    target=self._target_domain_name,
                    target_domain_metadata=self._target_domain_metadata,
                    prefix_path=self._base_path,
                    name=f"tester_{mapping_key}_{fold_id}_over{frac}",
                )

                transfer_candidate_dataset = {
                    "facts": train_target_data["facts"] + mapped_data["facts"],
                    "pos": train_target_data["pos"] + reinforced_pos,
                    "neg": train_target_data["neg"] + reinforced_neg,
                }

                tester_model.train(transfer_candidate_dataset)
                cand_train_metrics = tester_model.get_train_metrics()

                tester_model.test(validation_target_data)
                cand_val_metrics = tester_model.get_test_metrics()
                tester_model.delete()

                # Avalia o Oversampling usando a Trava e o Limiar Dinâmico
                if better(cand_val_metrics, cand_train_metrics, best_val, best_train):
                    best_val.update({
                        "auc_pr": cand_val_metrics["auc_pr"],
                        "auc_roc": cand_val_metrics["auc_roc"],
                        "choice": f"Full Source + {frac * 100}% Oversampling Elite",
                        "pos": reinforced_pos,
                        "neg": reinforced_neg,
                    })
                    best_train = cand_train_metrics.copy()

            selector_model.delete()

            # ---------- 5. FINALIZA O MAPEAMENTO ----------
            improved_over_baseline = best_val["choice"] != "Target-only"

            if improved_over_baseline:
                transfer_dataset[source_domain_name][mapping_key] = {
                    "facts": mapped_data["facts"],  # Fatos são sempre passados integralmente
                    "pos": best_val["pos"],
                    "neg": best_val["neg"],
                }
                print(
                    f"[{source_domain_name}] APROVADO: {best_val['choice']} | "
                    f"Val AUC-PR: {best_val['auc_pr']:.4f} (Treino protegido)"
                )
            else:
                print(
                    f"[{source_domain_name}] REJEITADO (Falhou na Trava de Treino ou Risco/Ganho): Mantendo Target-only. | "
                    f"Baseline Val AUC-PR: {no_transfer_val.get('auc_pr', 0):.4f}"
                )

        return transfer_dataset

    def _kg_training_workflow(self, fold_configuration):
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
            name="kg_model"
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
                    transfer_learning_raw_dataset = self._get_transfer_dataset(
                        fold_configuration=fold_configuration
                    )

                    if transfer_learning_raw_dataset is not None:
                        result["transfer_learning"] = True
                        self._dataset.process_source_from_data(transfer_learning_raw_dataset)

                    else:
                        tl_not_possible.append(fold_configuration["train"][0])
                        result["log"] += "Transfer-learning dataset could not be generated."

                tl_train_metrics, tl_test_metrics = self._kg_training_workflow(
                    fold_configuration=fold_configuration
                )

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
        #default_tl_out = self.default_transfer_learning()

        #results["default_transfer_learning_results"] = default_tl_out
        results["kg_transfer_learning_results"] = kg_tl_out
        
        # Salva os resultados em arquivo JSON
        self._save_results(results)
        
        return results