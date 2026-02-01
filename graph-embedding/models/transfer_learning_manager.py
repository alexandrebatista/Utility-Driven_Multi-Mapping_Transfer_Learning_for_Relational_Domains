import os
import json
from collections import defaultdict

from database.utils import graph_data_to_database, raw_data_to_database, convert_to_prolog_data, convert_datatomap_to_prologdata
from database.database import DatabaseWrapper
from mapping.apply_mappings import read_source_domain_and_apply_mapping
from mapping.utils import select_mappings
from .model import RdnBoost, load_model
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
        self._folds_split = self._dataset.generate_fold_splits()

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

    def get_knowledge_gap_data(self, fold_configuration):
        if self._best_model_was_not_found(fold_configuration["train"][0]):
            return None, None

        best_model = self._search_target_best_model(fold_configuration)

        if best_model is None:
            return None, None

        filtered_samples = best_model.process_data(min_fraction=0.01, score_threshold=0.5)

        knowledge_gap_data = {
            "pos": [item["raw"] for item in filtered_samples["pos_below"]],
            "neg": [item["raw"] for item in filtered_samples["neg_below"]],
            **self._dataset.get_target_facts(fold_configuration["train"]),
        }

        knowledge_data = {
            "pos": [item["raw"] for item in filtered_samples["pos_above"]],
            "neg": [item["raw"] for item in filtered_samples["neg_above"]],
            **self._dataset.get_target_facts(fold_configuration["train"]),
        }

        if len(knowledge_gap_data["pos"]) == 0 or len(knowledge_gap_data["neg"]) == 0:
            return None, knowledge_data

        return knowledge_gap_data, knowledge_data

    def _search_target_best_model(self, fold_configuration, save=True):
        from pathlib import Path

        name = f"best_model_{self._target_domain_name}_{fold_configuration['train'][0]}"
        path = Path(os.path.join(self._base_path, f"selector_models/pkl/", f"{name}.pkl"))

        if path.exists():
            return load_model(path)

        dataset_metadata = self._dataset.get_target_domain_metadata()

        train_dataset = self._dataset.generate_train_srldatabase(
            fold_configuration["train"],
            use_recursion=self._target_uses_recursion
        )

        model_settings = {"disable_limiters": True}

        number_trees_list = [10, 50, 100]
        max_tree_depth_list = [3, 8]

        for number_trees in number_trees_list:
            model_settings["nEstimators"] = number_trees

            for max_tree_depth in max_tree_depth_list:
                model_settings["maxTreeDepth"] = max_tree_depth

                model = RdnBoost(
                    target_domain_metadata=dataset_metadata,
                    model_settings=model_settings,
                    prefix_path=self._base_path,
                    name=name
                )

                model.train(train_dataset)

                pos_acc, neg_acc = model.get_train_accuracies(0.5)

                print(
                    f"{str(number_trees)}:{str(max_tree_depth)} - Positive Accuracy: {str(pos_acc)}. Negative Accuracy: {str(neg_acc)}")

                if pos_acc > 0.5 and neg_acc > 0.5:
                    if save:
                        model.save()
                    print("Model found!")
                    print(number_trees)
                    print(max_tree_depth)
                    self._update_best_models_found(fold_configuration['train'][0], True)
                    return model
                model.delete()

        self._update_best_models_found(fold_configuration['train'][0], False)
        return None

    def _get_transfer_dataset_2(self, fold_configuration):
        """
        Realiza Transfer Learning baseada em instâncias ("Instance-based Transfer").
        Usa RDN-Boost com estratégia Top-K Score para selecionar subgrafos.
        """

        import random
        import math
        import copy
        from collections import defaultdict

        # =========================================================================
        # 0. FUNÇÕES AUXILIARES
        # =========================================================================

        def extract_args(predicate_str):
            try:
                if "(" not in predicate_str: return []
                content = predicate_str.split("(")[1].replace(").", "").replace(")", "")
                return [x.strip() for x in content.split(",")]
            except IndexError:
                return []

        def extract_targets_by_position(fact_list, base_name="is_gap"):
            targets_by_pos = defaultdict(list)
            entities_by_pos = defaultdict(set)
            for fact in fact_list:
                args = extract_args(fact)
                for i, arg in enumerate(args):
                    entities_by_pos[i].add(arg)
            for i, entities in entities_by_pos.items():
                targets_by_pos[i] = [f"{base_name}_{i}({entity})." for entity in entities]
            return targets_by_pos, entities_by_pos

        def filter_relations_strict(relation_list, approved_dict):
            """Só aceita relação se args[i] estiver em approved_dict[i]"""
            selected_samples = []
            for relation in relation_list:
                args = extract_args(relation)
                if not args: continue
                is_fully_approved = True
                for i, arg in enumerate(args):
                    if i in approved_dict:
                        if arg not in approved_dict[i]:
                            is_fully_approved = False
                            break
                if is_fully_approved:
                    selected_samples.append(relation)
            return selected_samples

        def _inject_target_mode(metadata, new_predicate_name, pos_idx, ref_predicate_name):
            new_metadata = copy.deepcopy(metadata)
            ref_types = new_metadata.get('modes', {}).get(ref_predicate_name)
            inferred_type = ref_types[pos_idx] if (ref_types and pos_idx < len(ref_types)) else "thing"

            if 'modes' not in new_metadata: new_metadata['modes'] = {}
            new_metadata['modes'][new_predicate_name] = [inferred_type]

            if 'predicates' in new_metadata and new_predicate_name not in new_metadata['predicates']:
                new_metadata['predicates'].append(new_predicate_name)

            if 'raw_modes' in new_metadata:
                new_metadata['raw_modes'].append(f"{new_predicate_name}(+{inferred_type}).")

            new_metadata['target_predicate'] = new_predicate_name
            return new_metadata

        # =========================================================================
        # 1. CÁLCULO DE COTAS
        # =========================================================================

        # current_gap_pos = len(knowledge_gap_data["pos"])
        # current_know_pos = len(knowledge_data["pos"])
        # total_pos_current = current_gap_pos + current_know_pos
        # ratio = 0.5
        #
        # if total_pos_current > 0:
        #     numerator = (ratio * total_pos_current) - current_gap_pos
        #     denom = (1 - ratio)
        #     needed_pos = math.ceil(numerator / denom) if denom != 0 else 0
        #     needed_pos = max(0, needed_pos)
        # else:
        #     needed_pos = 10

        needed_pos = 1000

        needed_neg = needed_pos * 2
        total_needed = needed_pos + needed_neg

        knowledge_gap_data = {
            **self._dataset.get_target_facts(fold_configuration["train"]),
            **self._dataset.get_target_pos(fold_configuration["train"]),
            **self._dataset.get_target_neg(fold_configuration["train"]),
        }

        print(f"\n=== TRANSFERÊNCIA INICIADA ===")
        print(f"[Meta Global] Buscar {total_needed} entidades por posição (Ratio: {1}).")

        if needed_pos == 0:
            print("-> Meta atingida ou dataset vazio. Retornando vazio.")
            return defaultdict(dict)

        # =========================================================================
        # 2. EXECUÇÃO MULTI-MODELO
        # =========================================================================

        transfer_dataset = defaultdict(dict)
        target_predicate_name = self._target_domain_metadata.get('target_predicate', 'samevenue')

        # Acumuladores globais para o print final
        grand_total_pos = 0
        grand_total_neg = 0
        grand_total_facts = 0

        # Treino Positivo (Vem do GAP)
        gap_targets_raw = knowledge_gap_data["pos"] + knowledge_gap_data["neg"]
        gap_targets_dict, _ = extract_targets_by_position(gap_targets_raw, "is_gap")

        for mapping_key, mapped_data in self._source_domains.items():
            source_name = self._selected_mappings[mapping_key]["source"]
            print(f"\n--- [Processando Origem: {source_name} | Mapping ID: {mapping_key}] ---")

            final_transfer_facts = []
            final_transfer_pos = []
            final_transfer_neg = []

            source_targets_raw = mapped_data["pos"] + mapped_data["neg"]
            source_targets_dict, _ = extract_targets_by_position(source_targets_raw, "is_gap")

            active_positions = set(gap_targets_dict.keys())
            approved_by_pos = {}

            for pos_idx in active_positions:
                # 2.1 Preparação
                tool_train_pos = gap_targets_dict.get(pos_idx, [])
                all_source_targets = source_targets_dict.get(pos_idx, [])
                if not tool_train_pos or not all_source_targets: continue

                # Undersampling para treino
                tool_train_neg = random.sample(all_source_targets,
                                               min(len(all_source_targets), len(tool_train_pos) * 2))

                valid_predicates = [v for v in self._selected_mappings[mapping_key]["predicate_mappings"].values() if v]
                filtered_gap_facts = [f for f in knowledge_gap_data["facts"] if f.split('(', 1)[0] in valid_predicates]

                all_facts_train = (
                        filtered_gap_facts +
                        knowledge_gap_data["pos"] +
                        mapped_data["facts"] +
                        mapped_data["pos"]
                )

                unified_dataset = {"facts": all_facts_train, "pos": tool_train_pos, "neg": tool_train_neg}

                # 2.2 Treino
                current_target = f"is_gap_{pos_idx}"
                augmented_metadata = _inject_target_mode(self._target_domain_metadata, current_target, pos_idx,
                                                         target_predicate_name)

                selector_model = SelectorModel(
                    valid_predicates=valid_predicates,
                    target=current_target,
                    target_domain_metadata=augmented_metadata,
                    prefix_path=self._base_path
                )
                selector_model.train(unified_dataset, filter_predicates=False)

                # 2.3 Teste (Gera estimativas)
                test_facts = mapped_data["facts"] + mapped_data["pos"]
                mapped_data_for_test = {"facts": test_facts, "pos": all_source_targets, "neg": []}
                selector_model.test(mapped_data_for_test)

                # --- 2.4 SELEÇÃO VIA LOAD_ESTIMATES (Top-K) ---
                raw_estimates = selector_model.load_estimates()

                candidates_with_score = []
                for est in raw_estimates:
                    if est.get('predicate') == current_target:
                        entity = est['args'][0]
                        score = est['score']
                        candidates_with_score.append((entity, score))

                # Ordenação e Top-K
                candidates_with_score.sort(key=lambda x: x[1], reverse=True)
                top_k_candidates = candidates_with_score[:total_needed]

                approved_entities = set([x[0] for x in top_k_candidates])
                approved_by_pos[pos_idx] = approved_entities

                # Print de diagnóstico do modelo
                top_score = candidates_with_score[0][1] if candidates_with_score else 0.0
                print(
                    f"   > Argumento {pos_idx}: {len(approved_entities)} entidades selecionadas (Max Score: {top_score:.4f})")

            # 3. Filtragem Final
            batch_pos = filter_relations_strict(mapped_data["pos"], approved_by_pos)
            batch_neg = filter_relations_strict(mapped_data["neg"], approved_by_pos)
            final_transfer_pos.extend(batch_pos)
            final_transfer_neg.extend(batch_neg)

            # Fatos
            all_approved_flat = set()
            for s in approved_by_pos.values(): all_approved_flat.update(s)

            source_facts = mapped_data["facts"]
            batch_facts = []
            for fact in source_facts:
                args = extract_args(fact)
                if any(arg in all_approved_flat for arg in args):
                    batch_facts.append(fact)
            final_transfer_facts.extend(batch_facts)

            # Salva no dicionário de retorno
            transfer_dataset[source_name][mapping_key] = \
                {"facts": final_transfer_facts, "pos": final_transfer_pos, "neg": final_transfer_neg}

            # Acumula totais globais
            grand_total_pos += len(final_transfer_pos)
            grand_total_neg += len(final_transfer_neg)
            grand_total_facts += len(final_transfer_facts)

            # Print Resumo do Mapping atual
            print(f"   >>> Resultado Parcial ({source_name}):")
            print(f"       + {len(final_transfer_pos)} Amostras Positivas")
            print(f"       + {len(final_transfer_neg)} Amostras Negativas")
            print(f"       + {len(final_transfer_facts)} Fatos de Contexto")

        # Print Resumo Geral
        print(f"\n=== RESUMO FINAL DA TRANSFERÊNCIA ===")
        print(f"Total de Origens Processadas: {len(transfer_dataset)}")
        print(f"Total Geral Transferido:")
        print(f"  > Positivos: {grand_total_pos}")
        print(f"  > Negativos: {grand_total_neg}")
        print(f"  > Fatos:     {grand_total_facts}")
        print(f"=====================================\n")

        return transfer_dataset

    def _get_transfer_dataset(self, knowledge_gap_data, knowledge_data):
        import math

        transfer_dataset = defaultdict(dict)

        gap_pos_size = len(knowledge_gap_data["pos"])
        knowledge_pos_size = len(knowledge_data["pos"])

        total_pos = gap_pos_size + knowledge_pos_size

        ratio = 0.5
        negative_multiplier = 2  # 2 negativas para cada 1 positiva

        # 1. Calcula a necessidade da âncora (POSITIVOS) baseada no ratio de cobertura
        # Mantemos a fórmula original aqui para definir o "volume" de dados necessários
        needed_pos = (ratio * total_pos - gap_pos_size) / (1 - ratio)
        needed_pos = max(0, math.ceil(needed_pos))

        # 2. Calcula as negativas baseadas EXCLUSIVAMENTE na quantidade de positivas
        # Ignoramos o 'ratio' original para as negativas e obedecemos a regra 2:1
        needed_neg = needed_pos * negative_multiplier

        samples_added = {
            "pos": 0,
            "neg": 0
        }

        for mapping_key, mapped_data in self._source_domains.items():
            source_domain_name = self._selected_mappings[mapping_key]["source"]
            valid_predicates = [
                predicate for predicate in list(self._selected_mappings[mapping_key]["predicate_mappings"].values())
                if predicate is not None
            ]

            if samples_added["pos"] == needed_pos and samples_added["neg"] == needed_neg:
                break

            approach_1_failed = False
            if knowledge_gap_data is not None:
                selector_model = SelectorModel(
                    valid_predicates=valid_predicates,
                    target=self._target_domain_name,
                    target_domain_metadata=self._target_domain_metadata,
                    prefix_path=self._base_path
                )

                # Learn the knowledge gap patterns
                selector_model.train(knowledge_gap_data)

                if selector_model.is_reliable():
                    # Look for source domain samples whose patterns match knowledge gap ones.
                    selector_model.test(mapped_data)

                    pos_transfer_samples, neg_transfer_samples = selector_model.get_best_samples()

                    number_pos_samples_to_add = needed_pos - samples_added["pos"]
                    number_neg_samples_to_add = needed_neg - samples_added["neg"]

                    samples_added["pos"] += len(pos_transfer_samples[:number_pos_samples_to_add])
                    samples_added["neg"] += len(neg_transfer_samples[:number_neg_samples_to_add])

                    raw_transfer_dataset = {
                        "facts": mapped_data["facts"],
                        "pos": [f"{sample['raw']}." for sample in pos_transfer_samples[:number_pos_samples_to_add]],
                        "neg": [f"{sample['raw'][1:]}." for sample in neg_transfer_samples[:number_neg_samples_to_add]]
                    }

                    transfer_dataset[source_domain_name][mapping_key] = raw_transfer_dataset
                else:
                    approach_1_failed = True

                selector_model.delete()
            else:
                approach_1_failed = True

            if approach_1_failed:
                selector_model = SelectorModel(
                    valid_predicates=valid_predicates,
                    target=self._target_domain_name,
                    target_domain_metadata=self._target_domain_metadata,
                    prefix_path=self._base_path
                )

                # Learn the patterns in the mapped data
                selector_model.train(mapped_data)

                if selector_model.is_reliable():
                    pos_transfer_samples, neg_transfer_samples = selector_model.get_best_samples()

                    # Look for source domain samples whose patterns match knowledge gap ones.
                    selector_model.test(knowledge_gap_data)

                    if selector_model.is_reliable():
                        number_pos_samples_to_add = needed_pos - samples_added["pos"]
                        number_neg_samples_to_add = needed_neg - samples_added["neg"]

                        samples_added["pos"] += len(pos_transfer_samples[:number_pos_samples_to_add])
                        samples_added["neg"] += len(neg_transfer_samples[:number_neg_samples_to_add])

                        raw_transfer_dataset = {
                            "facts": mapped_data["facts"],
                            "pos": [f"{sample['raw']}." for sample in pos_transfer_samples[:number_pos_samples_to_add]],
                            "neg": [f"{sample['raw'][1:]}." for sample in neg_transfer_samples[:number_neg_samples_to_add]]
                        }

                        transfer_dataset[source_domain_name][mapping_key] = raw_transfer_dataset

                selector_model.delete()

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

    def kg_transfer_learning(self, rounds: int = 1):
        out = []
        tl_not_possible = []

        for fold_configuration in self._folds_split:
            for _ in range(rounds):
                self._dataset.reset_source_data()

                result = {"transfer_learning": False, "fold_configuration": fold_configuration, "log": ""}
                # Generate dataset with misclassified by the model

                if fold_configuration["train"][0] not in tl_not_possible:
                    # Search for a deficiency in the dataset
                    # knowledge_gap_data, knowledge_data = self.get_knowledge_gap_data(
                    #     fold_configuration=fold_configuration,
                    # )

                    # Get transfer dataset
                    #if knowledge_gap_data is not None:
                    transfer_learning_raw_dataset = self._get_transfer_dataset_2(
                        #knowledge_gap_data=knowledge_gap_data,
                        #knowledge_data=knowledge_data,
                        fold_configuration=fold_configuration
                    )

                    if transfer_learning_raw_dataset:
                        result["transfer_learning"] = True
                        self._dataset.process_source_from_data(transfer_learning_raw_dataset)

                    else:
                        tl_not_possible.append(fold_configuration["train"][0])
                        result["log"] += "Transfer-learning dataset could not be generated."
                    # else:
                    #     # Transfer learning is not suitable.
                    #     tl_not_possible.append(fold_configuration["train"][0])
                    #     result["log"] += "Best model not found."

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