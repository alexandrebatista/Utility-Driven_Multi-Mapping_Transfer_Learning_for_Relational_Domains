import json
import sys
import os
from models.transfer_learning_manager import TransferlearningManager
from mapping.utils import select_mappings

def run():
    # 1. Recebe o índice do job (0, 1, 2...)
    job_idx = int(sys.argv[1])
    
    # 2. Carrega a lista de mappings
    with open('mappings.json', 'r') as f:
        data = json.load(f)
    
    if job_idx >= len(data):
        return

    # 3. Extrai as informações e define o identificador
    config = data[job_idx]
    source_domain = config['source']
    target_domain = config['target']
    
    experiment_id = f"exp_{job_idx}_{source_domain}_to_{target_domain}"
    base_path = os.path.join("results", experiment_id)

    # --- CHECAGEM DE EXISTÊNCIA ---
    # Verifica se a pasta já existe e se não está vazia
    if os.path.exists(base_path) and len(os.listdir(base_path)) > 0:
        print(f"✅ Experimento {experiment_id} já concluído. Pulando...")
        return 
    # ------------------------------

    os.makedirs(base_path, exist_ok=True)
    print(f"🚀 Iniciando Job {job_idx}: {experiment_id}")

    # 4. Configura o Manager e executa
    tl_manager = TransferlearningManager(
        base_path=base_path,
        target_domain_name=target_domain
    )

    tl_manager.set_mappings({
        str(job_idx): {
            "predicate_mappings": config["relation_mapping"], 
            "type_mappings": config["term_type_mapping"],
            "source": config["source"],
            "target": config["target"],
            "id": str(job_idx)
        }
    })
    tl_manager.perform_transfer_learning()

if __name__ == "__main__":
    run()