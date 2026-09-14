from pathlib import Path


def create_folder(path):
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

def select_mappings(mappings_dict, source_domain=None, number_of_mappings_to_select=10, selection_strategy='random'):
    import random

    if selection_strategy == 'random':
        if source_domain is not None:
            mappings_considered = [key for key, value in mappings_dict.items() if value["source"] == source_domain]
        else:
            mappings_considered = list(mappings_dict.keys())

        # Corrigir a lógica: fazer sample SE houver mapeamentos suficientes
        selected_mappings_keys = random.sample(
            mappings_considered, 
            min(number_of_mappings_to_select, len(mappings_considered))
        )
    else:
        raise Exception('Invalid selection strategy: {}'.format(selection_strategy))

    return {key: value for key, value in mappings_dict.items() if key in selected_mappings_keys}