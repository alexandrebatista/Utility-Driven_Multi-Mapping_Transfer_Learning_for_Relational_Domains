import os
from pathlib import Path

from mapping.mapping import generate_mappings
from mapping.apply_mappings import apply_mappings

def create_folder(path):
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


# Se um predicado pred(A,B) é mapeado para o predicado alvo de destino, então eu preciso que existam predicados em facts.pl que utilizem os tipos A e B
# Juntar todos os folds
# Aplicar a metodologia da Thaís se o número de mapeamentos for maior que 10 (?)
# Gerar o neg.pl baseado nos objetos do tipo A e B que aparecem no facts.pl e/ou no pos.pl

# Talvez atualizar o mode.pl de destino para permitir recursividade

def generating_mappings_for(target_domain_name):
    base_dir = "./preprocessed"
    target_directory = f"{base_dir}/{target_domain_name}/fold00"
    output_path = f"/home/alexandre/mestrado/graph-embedding/mapped_datasets/to_{target_domain_name}"

    create_folder(output_path)

    print(f"to {target_domain_name}")

    path = Path(base_dir)
    for item in path.iterdir():
        if item.is_dir() and item.name != target_domain_name:
            source_path = os.path.join(item)
            source_directory = f"{source_path}/fold00"

            solutions = generate_mappings(source_directory, target_directory)
            #solutions.print_coherent_mappings()

            print(f"\t {item.name} : {len(solutions.get_robust_mappings())}")

            if len(solutions.get_robust_mappings()) < 10:
                create_folder(os.path.join(output_path, item.name))

                apply_mappings(source_path, os.path.join(output_path, item.name), solutions.get_robust_mappings(), solutions.get_target_target_predicate())

if __name__ == '__main__':
    path = Path("./preprocessed")
    generating_mappings_for("cora")
    generating_mappings_for("imdb")
    generating_mappings_for("nell_finances")
    generating_mappings_for("nell_sports")
    generating_mappings_for("twitter")
    generating_mappings_for("uwcse")
    generating_mappings_for("yeast")

