import os
import json
from collections import defaultdict

from database.utils import graph_data_to_database, raw_data_to_database, convert_to_prolog_data, convert_datatomap_to_prologdata
from database.database import DatabaseWrapper
from mapping.apply_mappings import read_source_domain_and_apply_mapping
from mapping.utils import select_mappings
from .model import RdnBoost, load_model
from .selector_model import SelectorModel, generate_selector_model


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

    import math
    import random

    def _get_transfer_dataset_2(self, knowledge_gap_data, knowledge_data):
        """
        Realiza Transfer Learning baseada em instâncias.
        Usa um modelo RDN-Boost como ferramenta para identificar subgrafos no Source
        que são estruturalmente similares ao Gap do Target.
        """

        import random
        import math

        # =========================================================================
        # 0. FUNÇÕES AUXILIARES INTERNAS
        # =========================================================================

        def extract_args(predicate_str):
            """
            Parser robusto.
            Entrada: 'samevenue(src_paper_1, src_venue_A)'
            Saída: ['src_paper_1', 'src_venue_A']
            """
            try:
                if "(" not in predicate_str: return []
                content = predicate_str.split("(")[1].replace(")", "")
                return [x.strip() for x in content.split(",")]
            except IndexError:
                return []

        def extract_entities_as_target(fact_list, predicate_name="is_gap"):
            """
            Transforma lista de relações em lista de alvos de entidade.
            Ex: ['samevenue(a,b)'] -> ['is_gap(a)', 'is_gap(b)']
            """
            entities = set()
            for fact in fact_list:
                args = extract_args(fact)
                for arg in args:
                    entities.add(arg)
            return [f"{predicate_name}({entity})" for entity in entities]

        def parse_entities_from_preds(pred_list):
            """
            Reverte a predição da ferramenta para IDs puros.
            Entrada: ['is_gap(src_a)', 'is_gap(src_b)'] -> Saída: {'src_a', 'src_b'}
            """
            entities = set()
            for pred in pred_list:
                args = extract_args(pred)
                if args:
                    entities.add(args[0])
            return entities

        def filter_relations_by_entities(relation_list, selected_entities, strict=True):
            """
            Reconstrói as amostras relacionais originais (samevenue).
            Strict=True: Só aceita se AMBOS os argumentos foram selecionados.
            """
            selected_samples = []
            selected_set = set(selected_entities)  # Set para busca O(1)

            for relation in relation_list:
                args = extract_args(relation)
                if len(args) < 2: continue

                arg1, arg2 = args[0], args[1]

                if strict:
                    if arg1 in selected_set and arg2 in selected_set:
                        selected_samples.append(relation)
                else:
                    if arg1 in selected_set or arg2 in selected_set:
                        selected_samples.append(relation)

            return selected_samples

        # =========================================================================
        # 1. CÁLCULO DE COTAS (RATIO 0.2 + REGRA 2:1)
        # =========================================================================

        current_gap_pos = len(knowledge_gap_data["pos"])
        current_know_pos = len(knowledge_data["pos"])
        total_pos_current = current_gap_pos + current_know_pos

        ratio = 0.2

        # Fórmula: x = (Ratio * Total - Gap) / (1 - Ratio)
        if total_pos_current > 0:
            numerator = (ratio * total_pos_current) - current_gap_pos
            denom = (1 - ratio)
            needed_pos = math.ceil(numerator / denom) if denom != 0 else 0
            needed_pos = max(0, needed_pos)
        else:
            needed_pos = 10  # Fallback se dataset for vazio

        needed_neg = needed_pos * 2

        print(f"[Transfer] Meta: Buscar {needed_pos} Positivos e {needed_neg} Negativos no Source.")

        if needed_pos == 0:
            return {"facts": [], "pos": [], "neg": []}

        # =========================================================================
        # 2. ITERAÇÃO E EXECUÇÃO DA FERRAMENTA
        # =========================================================================

        final_transfer_facts = []
        final_transfer_pos = []
        final_transfer_neg = []

        # O Treino POSITIVO da ferramenta são sempre as entidades do GAP
        # (Queremos aprender a estrutura delas)
        gap_targets_raw = knowledge_gap_data["pos"] + knowledge_gap_data["neg"]
        tool_train_pos = extract_entities_as_target(gap_targets_raw, "is_gap")

        for mapping_key, mapped_data in self._source_domains.items():
            valid_predicates = [
                predicate for predicate in list(self._selected_mappings[mapping_key]["predicate_mappings"].values())
                if predicate is not None
            ]

            # --- A. PREPARAÇÃO DO DATASET UNIFICADO ---

            # O Treino NEGATIVO da ferramenta é uma amostra do SOURCE
            source_targets_raw = mapped_data["pos"] + mapped_data["neg"]
            all_source_entities = extract_entities_as_target(source_targets_raw, "is_gap")

            # Undersampling para treino rápido do discriminador
            tool_train_neg = random.sample(
                all_source_entities,
                min(len(all_source_entities), len(tool_train_pos) * 2)
            )

            filtered_gap_data_facts = [fact for fact in knowledge_gap_data["facts"] if fact.split('(', 1)[0] in valid_predicates]

            # Fatos Unificados: O RDN precisa ver Source e Gap juntos
            all_facts = (
                    filtered_gap_data_facts +
                    knowledge_gap_data["pos"] +  # Alvos originais viram fatos de contexto
                    knowledge_gap_data["neg"] +
                    mapped_data["facts"] +
                    mapped_data["pos"] +
                    mapped_data["neg"]
            )

            unified_dataset = {
                "facts": all_facts,
                "pos": tool_train_pos,
                "neg": tool_train_neg
            }

            # --- B. TREINO E TESTE DO DISCRIMINADOR ---

            selector_model = SelectorModel(
                valid_predicates=valid_predicates,
                target=self._target_domain_name,
                target_domain_metadata=self._target_domain_metadata,
                prefix_path=self._base_path
            )

            # Treina para distinguir Gap vs Source
            selector_model.train(unified_dataset, filter_predicates=False)

            # Prepara teste: Queremos avaliar TODO o source
            # Injetamos as queries 'is_gap(X)' para todas as entidades source
            mapped_data_for_test = mapped_data.copy()

            # Nota: O 'test' do SelectorModel deve usar as queries implícitas ou as que passarmos.
            # Aqui assumimos que ele vai testar contra o que estiver definido no mapped_data
            # (que contem pos/neg originais).
            # Para garantir, poderíamos injetar 'pos': all_source_entities se o método test usar a chave 'pos'.
            selector_model.test(mapped_data_for_test)

            # --- C. SELEÇÃO E CORTES (SELECTION) ---

            # Recebe listas ['is_gap(src_a)', ...] que passaram no threshold (ex: > 0.5)
            # O modelo retorna separado quem era originalmente Pos ou Neg no source
            preds_pos_raw, preds_neg_raw = selector_model.get_best_samples()

            # Extrai IDs limpos
            candidates_pos = parse_entities_from_preds(preds_pos_raw)
            candidates_neg = parse_entities_from_preds(preds_neg_raw)

            # Aplica as COTAS (Se vier mais que o necessário, fazemos downsample)
            # Se vier menos (score baixo), pegamos o que tem.

            selected_pos_entities = set()
            if len(candidates_pos) > 0:
                k = min(len(candidates_pos), needed_pos)
                selected_pos_entities = set(random.sample(list(candidates_pos), k))

            selected_neg_entities = set()
            if len(candidates_neg) > 0:
                k = min(len(candidates_neg), needed_neg)
                selected_neg_entities = set(random.sample(list(candidates_neg), k))

            all_approved_entities = selected_pos_entities.union(selected_neg_entities)

            print(
                f"[Domain {mapping_key}] Selecionados {len(selected_pos_entities)} Pos, {len(selected_neg_entities)} Neg.")

            # --- D. FILTRAGEM DE RELAÇÕES E FATOS (TRANSFER) ---

            # 1. Recupera as relações samevenue(a,b)
            batch_pos = filter_relations_by_entities(mapped_data["pos"], all_approved_entities, strict=True)
            batch_neg = filter_relations_by_entities(mapped_data["neg"], all_approved_entities, strict=True)

            final_transfer_pos.extend(batch_pos)
            final_transfer_neg.extend(batch_neg)

            # 2. Recupera fatos de contexto (author, keyword, etc)
            # Trazemos fatos se eles mencionarem QUALQUER entidade aprovada
            source_facts = mapped_data["facts"]
            batch_facts = []
            for fact in source_facts:
                args = extract_args(fact)
                if any(arg in all_approved_entities for arg in args):
                    batch_facts.append(fact)

            final_transfer_facts.extend(batch_facts)

        print(
            f"[Final] Transferindo: {len(final_transfer_pos)} Pos, {len(final_transfer_neg)} Neg, {len(final_transfer_facts)} Fatos.")

        return {
            "facts": final_transfer_facts,
            "pos": final_transfer_pos,
            "neg": final_transfer_neg
        }

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
                selector_model = generate_selector_model(
                    base_path=self._base_path,
                    valid_predicates=valid_predicates,
                    dataset=knowledge_gap_data,
                    target_domain_name=self._target_domain_name,
                    target_domain_metadata=self._target_domain_metadata,
                )

                if selector_model:
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

    def kg_transfer_learning(self, rounds: int = 3):
        out = []
        tl_not_possible = []

        for fold_configuration in self._folds_split:
            for _ in range(rounds):
                self._dataset.reset_source_data()

                result = {"transfer_learning": False, "fold_configuration": fold_configuration, "log": ""}
                # Generate dataset with misclassified by the model

                if fold_configuration["train"][0] not in tl_not_possible:
                    # Search for a deficiency in the dataset
                    knowledge_gap_data, knowledge_data = self.get_knowledge_gap_data(
                        fold_configuration=fold_configuration,
                    )

                    # Get transfer dataset
                    if knowledge_gap_data is not None:
                        transfer_learning_raw_dataset = self._get_transfer_dataset(
                            knowledge_gap_data=knowledge_gap_data,
                            knowledge_data=knowledge_data
                        )

                        if transfer_learning_raw_dataset is not None:
                            result["transfer_learning"] = True
                            self._dataset.process_source_from_data(transfer_learning_raw_dataset)

                        else:
                            tl_not_possible.append(fold_configuration["train"][0])
                            result["log"] += "Transfer-learning dataset could not be generated."
                    else:
                        # Transfer learning is not suitable.
                        tl_not_possible.append(fold_configuration["train"][0])
                        result["log"] += "Best model not found."

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

        default_tl_out = self.default_transfer_learning()
        kg_tl_out = self.kg_transfer_learning()

        results["default_transfer_learning_results"] = default_tl_out
        results["kg_transfer_learning_results"] = kg_tl_out
        
        # Salva os resultados em arquivo JSON
        self._save_results(results)
        
        return results