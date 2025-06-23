import json
import os
from pathlib import Path

from tqdm import tqdm

from mapping.apply_mappings import read_domain
from mapping.mapping import generate_mappings, parse_modes_file
from mapping.metrics import compare_graphs, extract_graph_metrics

from joblib import Parallel, delayed

from models.selector_model import SelectorModel

# Variáveis globais
_source_domain_data = None
_target_domain_modes = None
_target_metrics = None

BASE_DIR = "/home/alexandre/mestrado/graph-embedding/preprocessed"


def compute_metrics(index, solution, source_domain_data, target_domain_modes, target_metrics, source_domain_path,
                    target_domain_path):
    source_domain_mapped_data = generate_source_domain_mapped_data(
        solution,
        source_domain_data,
        list(target_domain_modes.keys())
    )

    source_domain_mapped_metrics = extract_graph_metrics(source_domain_mapped_data, target_domain_modes)
    kl_divergences = (target_metrics, source_domain_mapped_metrics)

    solution["kl_divergences"] = kl_divergences
    solution["metrics"] = source_domain_mapped_metrics
    solution["source"] = source_domain_path.name
    solution["target"] = target_domain_path.name
    solution["id"] = index

    return solution


def process_all_mappings(mappings, source_domain_path, target_domain_path):
    all_mappings = list(enumerate(mappings))

    source_domain_data = read_domain(source_domain_path)
    domain_data = read_domain(target_domain_path)

    target_directory = f"{BASE_DIR}/{target_domain_path.name}/fold00"
    target_domain_modes = parse_modes_file(os.path.join(target_directory, "modes.pl"))

    # tqdm no loop do Parallel
    results = Parallel(n_jobs=-1)(
        delayed(compute_metrics)(
            index, mapping, source_domain_data, target_domain_modes,
            target_metrics, source_domain_path, target_domain_path
        )
        for index, mapping in tqdm(all_mappings, desc="Calculando métricas", unit="map")
    )

    return sorted(results, key=lambda x: x["mean_ranking_kl"])


def save_mappings(path, mappings):
    with open(os.path.join(path, "mappings.json"), 'w') as f:
        json.dump(mappings, f)


def create_folder(path):
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def generate_source_domain_mapped_data(
        mapping,
        source_domain_data,
        target_predicates,
):
    # Cópia dos dados
    mapped_source_domain_data = {predicate: set() for predicate in target_predicates}

    for predicate, args_instances in source_domain_data.items():
        if mapping["predicate_mappings"][predicate] is not None:
            mapped_source_domain_data[mapping["predicate_mappings"][predicate]].update(args_instances)

    return mapped_source_domain_data


def mappings_workflow(train_folds="fold00"):
    path = Path(BASE_DIR)
    for target_domain_path in path.iterdir():
        if target_domain_path.name == "uwcse":
            continue

        target_directory = f"{BASE_DIR}/{target_domain_path.name}/fold00"
        output_path = f"/home/alexandre/mestrado/graph-embedding/mapped_datasets/to_{target_domain_path.name}"
        create_folder(output_path)

        print(f"to {target_domain_path.name}")

        mapping_id = 0
        mappings_by_target = {}

        for source_domain_path in path.iterdir():
            if source_domain_path.name == target_domain_path.name or not source_domain_path.is_dir():
                continue

            source_path = os.path.join(source_domain_path)
            source_directory = f"{source_path}/fold00"

            mappings_path = Path(os.path.join(output_path, "mappings.json"))

            if mappings_path.exists():
                pass
            else:
                mappings = generate_mappings(source_domain_name=source_domain_path.name,
                                             source_dir=source_directory,
                                             target_domain_name=target_domain_path.name,
                                             target_dir=target_directory)

                print(f"\t {source_domain_path.name} : {len(mappings)}")

                #processed_mappings = process_all_mappings(mappings, source_domain_path, target_domain_path)

                for mapping in mappings:
                    mappings_by_target[mapping_id] = mapping
                    mapping_id += 1

        save_mappings(os.path.join(output_path), mappings_by_target)
