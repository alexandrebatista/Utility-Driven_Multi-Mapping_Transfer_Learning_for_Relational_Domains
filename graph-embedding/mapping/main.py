import json
import os
from pathlib import Path

from tqdm import tqdm

from mapping.apply_mappings import read_domain
from mapping.mapping import generate_mappings, parse_modes_file
from mapping.metrics import compare_graphs, extract_graph_metrics
from mapping.utils import TopKMapper
from mapping.apply_mappings import apply_mappings

from joblib import Parallel, delayed

# Variáveis globais
_source_domain_data = None
_target_domain_modes = None
_target_metrics = None


def compute_metrics(index, solution, source_domain_data, target_domain_modes, target_metrics, source_domain_path,
                    target_domain_path):
    source_domain_mapped_data = generate_source_domain_mapped_data(
        solution,
        source_domain_data,
        list(target_domain_modes.keys())
    )

    source_domain_mapped_metrics = extract_graph_metrics(source_domain_mapped_data, target_domain_modes)
    kl_divergences = compare_graphs(target_metrics, source_domain_mapped_metrics)

    solution["kl_divergences"] = kl_divergences
    solution["metrics"] = source_domain_mapped_metrics
    solution["source"] = source_domain_path.name
    solution["target"] = target_domain_path.name
    solution["id"] = index

    return kl_divergences, index, solution


def process_all_mappings(solutions, source_domain_data, target_domain_modes, target_metrics, source_domain_path,
                         target_domain_path):
    mappings_heap_kl = {
        "node_kl_ranking": TopKMapper(K=len(solutions.get_robust_mappings()), mode="min"),
        "edge_kl_ranking": TopKMapper(K=len(solutions.get_robust_mappings()), mode="min"),
        "degree_kl_in_kl_ranking": {predicate: TopKMapper(K=len(solutions.get_robust_mappings()), mode="min") for
                                    predicate in solutions.get_target_types()},
        "degree_kl_out_kl_ranking": {predicate: TopKMapper(K=len(solutions.get_robust_mappings()), mode="min") for
                                     predicate in solutions.get_target_types()},
    }

    all_solutions = list(enumerate(solutions.get_robust_mappings()))

    # tqdm no loop do Parallel
    results = Parallel(n_jobs=-1)(
        delayed(compute_metrics)(
            index, solution, source_domain_data, target_domain_modes,
            target_metrics, source_domain_path, target_domain_path
        )
        for index, solution in tqdm(all_solutions, desc="Calculando métricas", unit="map")
    )

    for kl_divergences, index, solution in results:
        mappings_heap_kl["node_kl_ranking"].update(kl_divergences["node_kl"], mapping_id=index, mapping_data=solution)
        mappings_heap_kl["edge_kl_ranking"].update(kl_divergences["edge_kl"], mapping_id=index, mapping_data=solution)

        for node_type in solutions.get_target_types():
            mappings_heap_kl["degree_kl_in_kl_ranking"][node_type].update(kl_divergences["degree_kl_in"][node_type],
                                                                          mapping_id=index,
                                                                          mapping_data=solution)

            mappings_heap_kl["degree_kl_out_kl_ranking"][node_type].update(kl_divergences["degree_kl_in"][node_type],
                                                                           mapping_id=index,
                                                                           mapping_data=solution)

    return mappings_heap_kl

def save_rankings(path, mappings_heap_kl):
    with open(os.path.join(path, "ranking_node_kl.json"), 'w') as f:
        json.dump(mappings_heap_kl["node_kl_ranking"].get_mappings(), f)

    with open(os.path.join(path, "ranking_edge_kl.json"), 'w') as f:
        json.dump(mappings_heap_kl["edge_kl_ranking"].get_mappings(), f)

    for node_type in mappings_heap_kl["degree_kl_in_kl_ranking"]:

        with open(os.path.join(path, f"ranking_{node_type}_in_kl.json"), 'w') as f:
            json.dump(mappings_heap_kl["degree_kl_in_kl_ranking"][node_type].get_mappings(), f)

        with open(os.path.join(path, f"ranking_{node_type}_out_kl.json"), 'w') as f:
            json.dump(mappings_heap_kl["degree_kl_out_kl_ranking"][node_type].get_mappings(), f)


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
    base_dir = "/home/alexandre/mestrado/graph-embedding/preprocessed"

    path = Path(base_dir)
    for target_domain_path in path.iterdir():
        if target_domain_path.name == "uwcse":
            continue

        target_directory = f"{base_dir}/{target_domain_path.name}/fold00"
        output_path = f"/home/alexandre/mestrado/graph-embedding/mapped_datasets/to_{target_domain_path.name}"
        create_folder(output_path)

        domain_data = read_domain(target_domain_path)
        target_domain_modes = parse_modes_file(os.path.join(target_directory, "modes.pl"))

        target_metrics = extract_graph_metrics(domain_data, target_domain_modes)

        with open(os.path.join(output_path, "metrics.json"), 'w') as f:
            json.dump(target_metrics, f)

        print(f"to {target_domain_path.name}")

        for source_domain_path in path.iterdir():
            if source_domain_path.name == target_domain_path.name or not source_domain_path.is_dir():
                continue

            source_path = os.path.join(source_domain_path)
            source_directory = f"{source_path}/fold00"

            solutions = generate_mappings(source_directory, target_directory)

            print(f"\t {source_domain_path.name} : {len(solutions.get_robust_mappings())}")

            source_domain_data = read_domain(source_path)

            mappings_heap_kl = process_all_mappings(solutions, source_domain_data, target_domain_modes, target_metrics,
                                                 source_domain_path, target_domain_path)

            create_folder(os.path.join(output_path, source_domain_path.name, "rankings"))

            save_rankings(path=os.path.join(output_path, source_domain_path.name, "rankings"), mappings_heap_kl=mappings_heap_kl)
            save_mappings(os.path.join(output_path, source_domain_path.name), solutions.get_robust_mappings())

            # create_folder(os.path.join(output_path, "best", source_domain_path.name))
            # create_folder(os.path.join(output_path, "worst", source_domain_path.name))
            #
            # apply_mappings(source_path,
            #                os.path.join(output_path, "best", source_domain_path.name),
            #                [item[2] for item in best_mappings],
            #                solutions.get_target_target_predicate(),
            #                solutions.get_target_modes())
