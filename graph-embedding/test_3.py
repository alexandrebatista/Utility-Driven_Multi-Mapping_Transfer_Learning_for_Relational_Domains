from models.transfer_learning_manager import TransferlearningManager
from database.utils import get_dataset_names


base_path = "./tl_experiments_2"

domains = get_dataset_names("./preprocessed")
targets = ["cora", "nell_finances", "yeast"]
target_domains = [{"domain": domain} for domain in targets if domain != "uwcse"]

for target_domain_name in target_domains:
    tl_manager = TransferlearningManager(
        base_path=base_path,
        target_domain_name=target_domain_name
    )

    for source_domain_name in domains:
        if source_domain_name == target_domain_name:
            continue

        tl_manager.select_mappings(source_domain_name=source_domain_name, number_of_mappings=5)






