import json

#from mapping.main import mappings_workflow
from models.main import models_workflow
#from models.selector_models_workflow import selector_models_workflow


def process_mappings(target_domain: str, source_domains: list, quality: str):
    paths = []
    for source_domain in source_domains:
        paths.append(f"./mapped_datasets/to_{target_domain}/{quality}/{source_domain}")

    out = []
    for mapping_folder in os.listdir(path):
        with open(os.path.join(path, mapping_folder), 'r', encoding='utf-8') as f:
            mappings = json.load(f)

        mapping_metadata = next((item for item in mappings if item.get('id') == mapping_folder), None)

        if os.path.isdir(os.path.join(path, mapping_folder)):
            models_workflow(
                target_domain=target_domain,
                train_folds=["fold00"],
                source_paths=[os.path.join(path, mapping_folder)],
                quality="best"
            )
            os.rename("./bsrl_data/data0", f"./bsrl_data/{source_domain} -> {target_domain} (mapping)")
            out.append({"source": source_domain,
                        "target": target_domain,
                        "mapping": mapping_folder,
                        "mean_ranking_kl": mapping_metadata["mean_ranking_kl"]})

    with open(f"./results/{target_domain}_{quality}", 'w', encoding='utf-8') as file:
        json.dump(out, file, ensure_ascii=False, indent=2)

def process_mapping(target_domain: str, source_domain: str, quality: str):
    path = f"./mapped_datasets/to_{target_domain}/{quality}/{source_domain}"

    with open(os.path.join(path, "mappings.json"), 'r', encoding='utf-8') as f:
        mappings = json.load(f)

    tmp = {}
    for mapping in mappings:
        tmp[mapping["id"]] = mapping
    mappings = tmp

    out = []
    for mapping_id in os.listdir(path):
        if os.path.isdir(os.path.join(path, mapping_id)):
            source_domains = [{
                "source_domain": source_domain,
                "mappings": {mapping_id: f"{path}/{mapping_id}"}
            }]

            models_workflow(
                target_domain=target_domain,
                train_folds=["fold00"],
                source_domains=source_domains,
                quality="best"
            )
            new_name = f"./bsrl_data/{source_domain} -> {target_domain} ({mapping_id})"
            if not os.path.exists(new_name):
                os.rename("./bsrl_data/data0", new_name)
                out.append({"source": source_domain,
                            "target": target_domain,
                            "mapping": mapping_id,
                            "mean_ranking_kl": mappings[int(mapping_id)]["mean_ranking_kl"]})

    with open(f"./results/{target_domain}_{quality}.json", 'w', encoding='utf-8') as file:
        json.dump(out, file, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    import os

    os.nice(10)

    data_path = "/home/alexandre/mestrado/graph-embedding/preprocessed"
    mappings_path = "/home/alexandre/mestrado/graph-embedding/mappings/"
    selector_models_path = "/home/alexandre/mestrado/graph-embedding/selector_models/"

    # ------------------------------------------------------------------------------------------------------------------
    # Generating Mappings and saving files

    #mappings_workflow()

    #selector_models_workflow(data_path, mappings_path, selector_models_path)

    # ------------------------------------------------------------------------------------------------------------------
    # Transforming the datasets in Srlearn databases

    # process_mapping(target_domain="cora", source_domain="imdb", quality="best")
    # process_mapping(target_domain="cora", source_domain="imdb", quality="worst")

    # models_workflow(
    #     target_domain="cora",
    #     train_folds=["fold00"],
    #     source_domains=["imdb"],
    #     quality="best"
    # )
    #
    # models_workflow(
    #     target_domain="cora",
    #     train_folds=["fold00"],
    #     source_domains=["imdb"],
    #     quality="worst"
    # )

    models_workflow(
        target_domain="cora",
    )
