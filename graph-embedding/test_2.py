from models.main import tl_experiment_1
from database.utils import get_dataset_names


base_path = "./tl_experiments"

model_path = "./tl_experiments"

targets = get_dataset_names("./preprocessed")
no_recursion_datasets = ["cora", "imdb", "uwcse"]
target_domains = [{"domain": domain, "use_recursion": domain not in no_recursion_datasets} for domain in targets if domain != "uwcse"]

for target_domain in target_domains:
    tl_experiment_1(
        base_path=base_path,
        model_path=model_path,
        target_domain=target_domain["domain"],
        rounds=3,
        use_recursion=target_domain["use_recursion"]
    )


