import json
import sys
import os
from models.transfer_learning_manager import TransferlearningManager
from mapping.utils import select_mappings


def run_single_experiment(job_index):
    # 1. Carregar os dados enviados pelo HTCondor
    with open('mappings.json', 'r') as f:
        mappings_list = json.load(f)

    if job_index >= len(mappings_list):
        return

    config = mappings_list[job_index]
    source_domain_name = config['source']
    target_domain_name = config['target']

    # Filtro de domínio (conforme seu original)
    if source_domain_name == target_domain_name or source_domain_name != "nell_sports":
        print(f"Pulando {source_domain_name}...")
        return

    # O identificador único do experimento para o base_path
    # Exemplo: ./experiments/job_0_nell_sports_to_cora
    experiment_id = f"job_{job_index}_{source_domain_name}_to_{target_domain_name}"
    base_path = os.path.join("./results", experiment_id)

    os.makedirs(base_path, exist_ok=True)

    # 2. Inicializar o Manager com o path único
    tl_manager = TransferlearningManager(
        base_path=base_path,
        target_domain_name=target_domain_name
    )

    # 3. Preparar e executar o mapeamento específico
    # Recriamos o formato que o seu select_mappings espera
    mapping_dict_input = {config.get('mapping_key', 'default'): config['relation_mapping']}

    selected_mappings = select_mappings(
        mappings_dict=mapping_dict_input,
        source_domain=source_domain_name,
        number_of_mappings_to_select=1
    )

    for mapping_key, mapping in selected_mappings.items():
        tl_manager.set_mappings({mapping_key: mapping})
        tl_manager.perform_transfer_learning()


if __name__ == "__main__":
    idx = int(sys.argv[1])
    run_single_experiment(idx)