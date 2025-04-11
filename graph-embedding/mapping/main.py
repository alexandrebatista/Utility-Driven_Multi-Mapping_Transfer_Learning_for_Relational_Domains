import os
from pathlib import Path

from mapping.apply_mappings import read_domain
from mapping.mapping import generate_mappings, parse_modes_file
from mapping.metrics import cosine_sim, extract_vector_metrics_from_domain_data, extract_graph_metrics
from mapping.utils import TopNMapper
from mapping.apply_mappings import apply_mappings
from tqdm import tqdm
import random



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
        target_domain_vec = extract_vector_metrics_from_domain_data(domain_data, target_domain_modes)

        mapping_id_counter = 0

        for source_domain_path in path.iterdir():
            if source_domain_path.name == target_domain_path.name or not source_domain_path.is_dir():
                continue

            source_path = os.path.join(source_domain_path)
            source_directory = f"{source_path}/fold00"

            solutions = generate_mappings(source_directory, target_directory)

            print(f"\t {source_domain_path.name} : {len(solutions.get_robust_mappings())}")

            source_domain_data = read_domain(source_path)
            top_mapper = TopNMapper(N=3)

            shuffled = random.sample(list(solutions.get_robust_mappings()), k=len(solutions.get_robust_mappings()))  # shuffle com index

            counter = 0
            threshold = 1000000
            for index, solution in tqdm(enumerate(shuffled)):
                source_domain_mapped_data = generate_source_domain_mapped_data(
                    solution,
                    source_domain_data,
                    list(target_domain_modes.keys())
                )

                source_domain_mapped_vec = extract_vector_metrics_from_domain_data(source_domain_mapped_data, target_domain_modes)

                similarity = cosine_sim(target_domain_vec, source_domain_mapped_vec)
                solution["cos_similarity"] = similarity.item()
                solution["source"] = source_domain_path.name
                solution["target"] = target_domain_path.name
                solution["id"] = mapping_id_counter
                mapping_id_counter += 1

                top_mapper.update(similarity, mapping_id=index, mapping_data=solution)
                counter += 1

                if counter > threshold:
                    break

            top_mappings = top_mapper.get_top_mappings()
            create_folder(os.path.join(output_path, source_domain_path.name))
            apply_mappings(source_path,
                           os.path.join(output_path, source_domain_path.name),
                           [item[2] for item in top_mappings],
                           solutions.get_target_target_predicate(),
                           solutions.get_target_modes())

            # extract_vector_metrics_from_domain_file("./preprocessed/cora")

            print(f"to {target_domain_path.name}")
