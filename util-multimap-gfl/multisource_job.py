import argparse
import json
import os
import sys
from models.transfer_learning_manager import TransferlearningManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--ids", required=True)  # Recebe string formatada: "0,1,2"
    args = parser.parse_args()

    # --- 1. VALIDAÇÃO DE SEGURANÇA (TRAVA DO TWITTER) ---
    if args.target.lower() == "twitter":
        print(f"❌ Abortando: {args.target} não possui folds suficientes para ser Target.")
        sys.exit(0)  # Encerra graciosamente para não gerar erro no Condor

    # --- 2. CARREGAMENTO DOS MAPEAMENTOS ORIGINAIS ---
    if not os.path.exists('mappings.json'):
        print("❌ Erro catastrófico: mappings.json não foi transferido para o node.")
        sys.exit(1)

    with open('mappings.json', 'r') as f:
        all_mappings = json.load(f)

    # --- 3. RECONSTRUÇÃO DO DICIONÁRIO NO FORMATO EXIGIDO PELO MANAGER ---
    target_ids = args.ids.split(",")
    relevant_mappings = {}
    
    for tid in target_ids:
        idx = int(tid)
        if idx >= len(all_mappings):
            print(f"⚠️ Índice {idx} fora dos limites do mappings.json (Tamanho: {len(all_mappings)})")
            continue

        config = all_mappings[idx]

        idx = str(idx)
        # Traduz a estrutura da lista original para as chaves internas do Manager

        relevant_mappings[idx] = {
            "predicate_mappings": config["relation_mapping"], 
            "type_mappings": config["term_type_mapping"],
            "source": config["source"],
            "target": config["target"],
            "id": idx
        }

    if not relevant_mappings:
        print("❌ Erro: Nenhum mapeamento válido foi extraído para este lote.")
        sys.exit(1)

    # --- 4. ESTRUTURA DE PASTAS HIERÁRQUICA E IDEMPOTÊNCIA ---
    # results/imdb_to_cora/size_2/gamma_0.2/ids_0-1/
    experiment_dir = f"ids_{args.ids.replace(',', '-')}"
    base_path = os.path.join(
        os.getcwd(), 
        "results", 
        f"{args.source}_to_{args.target}", 
        f"size_{args.size}",
        f"gamma_{args.gamma}",
        experiment_dir
    )

    if os.path.exists(base_path) and any(os.scandir(base_path)):
        print(f"✅ Experimento já concluído em: {experiment_dir}. Pulando...")
        return 

    os.makedirs(base_path, exist_ok=True)
    print(f"🚀 Iniciando Par {args.source}->{args.target} | Tamanho: {args.size} | Gamma: {args.gamma}")

    # --- 5. INICIALIZAÇÃO DO MANAGER E EXECUÇÃO ---
    tl_manager = TransferlearningManager(base_path=base_path, target_domain_name=args.target)
    tl_manager.set_gamma(args.gamma)
    tl_manager.set_mappings(relevant_mappings)
    
    try:
        tl_manager.perform_transfer_learning()
        print(f"✔️ Experimento {experiment_dir} finalizado com sucesso!")
    except Exception as e:
        print(f"❌ Erro crítico durante a execução do Transfer Learning: {str(e)}")
        raise e  # Repassa o erro para o Condor registrar no arquivo .err

if __name__ == "__main__":
    main()