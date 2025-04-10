import os
from copy import deepcopy
from pathlib import Path

from mapping.apply_mappings import read_domain
from mapping.mapping import generate_mappings, parse_modes_file
from mapping.metrics import extract_vector_metrics_from_domain_file, \
    cosine_sim, extract_vector_metrics_from_domain_data, extract_graph_metrics
from mapping.utils import TopNMapper
from mapping.apply_mappings import apply_mappings
from tqdm import tqdm




def create_folder(path):
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def generate_source_domain_mapped_data(
        mapping,
        source_domain_data,
        target_predicates,
        target_nodes
):
    source_domain_mapped_data = deepcopy(source_domain_data)
    for source_predicate, target_predicate in mapping["predicate_mappings"].items():
        if target_predicate is None:
            source_domain_mapped_data["edges"].pop(source_predicate)
        else:
            if target_predicate not in source_domain_mapped_data["edges"]:
                source_domain_mapped_data["edges"][target_predicate] = source_domain_mapped_data["edges"].pop(
                    source_predicate)
            else:
                source_domain_mapped_data["edges"][target_predicate].update(
                    source_domain_mapped_data["edges"].pop(source_predicate))

    for source_type, target_type in mapping["type_mappings"].items():
        if target_type is None:
            source_domain_mapped_data["nodes"].pop(source_type)
        else:
            if target_type not in source_domain_mapped_data["nodes"]:
                source_domain_mapped_data["nodes"][target_type] = source_domain_mapped_data["nodes"].pop(source_type)
            else:
                source_domain_mapped_data["nodes"][target_type].update(
                    source_domain_mapped_data["nodes"].pop(source_type))

    for predicate in target_predicates:
        if predicate not in source_domain_mapped_data["edges"]:
            source_domain_mapped_data["edges"][predicate] = set()

    for node in target_nodes:
        if node not in source_domain_mapped_data["nodes"]:
            source_domain_mapped_data["nodes"][node] = set()

    keys = list(source_domain_mapped_data["nodes"])
    for node in keys:
        if node not in target_nodes:
            source_domain_mapped_data["nodes"].pop(node)

    return source_domain_mapped_data


def mappings_workflow(train_folds="fold00"):
    base_dir = "/home/alexandre/mestrado/graph-embedding/preprocessed"

    path = Path(base_dir)
    for target_domain_path in path.iterdir():
        target_directory = f"{base_dir}/{target_domain_path.name}/fold00"
        output_path = f"/home/alexandre/mestrado/graph-embedding/mapped_datasets/to_{target_domain_path.name}"
        create_folder(output_path)

        target_modes = parse_modes_file(os.path.join(target_directory, "modes.pl"))
        target_domain_vec = extract_vector_metrics_from_domain_file(target_domain_path)

        for source_domain_path in path.iterdir():
            if source_domain_path.name == target_domain_path.name or not source_domain_path.is_dir():
                continue

            source_path = os.path.join(source_domain_path)
            source_directory = f"{source_path}/fold00"

            solutions = generate_mappings(source_directory, target_directory)

            print(f"\t {source_domain_path.name} : {len(solutions.get_robust_mappings())}")


            target_predicates = solutions.get_target_predicates()
            target_nodes = solutions.get_target_types()

            source_domain_data = read_domain(source_path)
            top_mapper = TopNMapper(N=3)

            for index, solution in tqdm(enumerate(solutions.get_robust_mappings())):
                source_domain_mapped_data = generate_source_domain_mapped_data(
                    solution,
                    source_domain_data,
                    target_predicates,
                    target_nodes
                )

                source_domain_mapped_vec = extract_vector_metrics_from_domain_data(source_domain_mapped_data)

                similarity = cosine_sim(target_domain_vec, source_domain_mapped_vec)

                top_mapper.update(similarity, mapping_id=index, mapping_data=solution)

            top_mappings = top_mapper.get_top_mappings()
            create_folder(os.path.join(output_path, source_domain_path.name))
            apply_mappings(source_path,
                           os.path.join(output_path, source_domain_path.name),
                           [item[2] for item in top_mappings],
                           solutions.get_target_target_predicate(),
                           solutions.get_target_modes())

            # extract_vector_metrics_from_domain_file("./preprocessed/cora")

            print(f"to {target_domain_path.name}")
