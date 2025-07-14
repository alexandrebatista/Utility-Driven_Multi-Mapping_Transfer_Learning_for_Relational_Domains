import json
import os
from database.utils import extract_domain_metadata, read_target_domain
from mapping.apply_mappings import read_source_domain_and_apply_mapping
from mapping.utils import create_folder
from models.selector_model import SelectorModel
from models.utils import set_to_uuid, save_to_pickle, load_pickle


def _get_models_metadata(metadata_path):
    if not os.path.exists(metadata_path):
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({}, f)

        return {}

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    return metadata


def _search_model_metadata(metadata, predicates):
    model_uuid = set_to_uuid(predicates)

    return metadata.get(model_uuid)


def _update_model_metadata(metadata_path, metadata):
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)


def selector_models_workflow(
        data_path,
        mappings_path,
        selector_models_path):
    models_path = os.path.join(selector_models_path, "models")
    models_metadata_path_dir = os.path.join(selector_models_path, "metadata")

    create_folder(models_path)
    create_folder(models_metadata_path_dir)

    for mappings_file in os.listdir(mappings_path):
        if mappings_file.endswith('.json') and mappings_file == "cora.json":
            file_path = os.path.join(mappings_path, mappings_file)
            print(f'  Reading file: {mappings_file}')

            # Open and load the JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                mappings = json.load(f)

            for mapping_id, mapping in mappings.items():
                # if mapping["source"] != "imdb":
                #     continue

                predicates = set(
                    [predicate for predicate in mapping["predicate_mappings"].values() if predicate is not None])

                model_metadata_path_file = os.path.join(models_metadata_path_dir, f"{mapping["target"]}.json")

                models_metadata = _get_models_metadata(model_metadata_path_file)
                model_metadata = _search_model_metadata(models_metadata, predicates)

                target_domain_path = os.path.join(data_path, mapping["target"])
                create_folder(os.path.join(models_path, mapping["target"]))
                target_domain_metadata = extract_domain_metadata(target_domain_path)

                if model_metadata is None:
                    selector_model = SelectorModel(valid_predicates=predicates,
                                                   source=mapping["source"],
                                                   target=mapping["target"],
                                                   target_domain_metadata=target_domain_metadata)

                    target_domain_data = read_target_domain(target_domain_path, folds=["fold00"])
                    selector_model.train(target_domain_data)

                    # Create model
                    models_metadata[set_to_uuid(predicates)] = selector_model.get_metadata()
                    _update_model_metadata(model_metadata_path_file, models_metadata)
                    save_to_pickle(os.path.join(models_path, mapping["target"], set_to_uuid(predicates)),
                                   selector_model)
                else:
                    # load model
                    selector_model = load_pickle(os.path.join(models_path, mapping["target"], set_to_uuid(predicates)))

                # read source domain
                source_domain_path = os.path.join(data_path, mapping["source"])

                source_domain_mapped_data = read_source_domain_and_apply_mapping(source_domain_path,
                                                                                 mapping,
                                                                                 target_domain_metadata)

                selector_model.test(source_domain_mapped_data)

                mapping["train_metrics"] = selector_model.get_train_metrics()
                mapping["test_metrics"] = selector_model.get_test_metrics()

            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(mappings, file)
