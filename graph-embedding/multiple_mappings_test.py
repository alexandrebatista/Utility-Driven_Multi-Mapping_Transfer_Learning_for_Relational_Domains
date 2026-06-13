import os
import json

from models.transfer_learning_manager import TransferlearningManager
from database.utils import get_dataset_names
from mapping.utils import select_mappings


base_path = "./tl_experiments_3"

domains = get_dataset_names("./preprocessed")
targets = ["cora"]#, "nell_finances", "yeast"]
target_domains = [domain for domain in targets if domain != "uwcse"]


for target_domain_name in target_domains:
    with open(os.path.join("./mappings", f"{target_domain_name}.json"), "r", encoding="utf-8") as f:
        mappings = json.load(f)

    tl_manager = TransferlearningManager(
        base_path=base_path,
        target_domain_name=target_domain_name
    )

    for source_domain_name in domains:
        if source_domain_name == target_domain_name or source_domain_name != "imdb":
            continue

        selected_mappings = select_mappings(
            mappings_dict=mappings,
            source_domain=source_domain_name,
            number_of_mappings_to_select=5
        )

        tl_manager.set_mappings(selected_mappings)
        tl_manager.perform_transfer_learning()

