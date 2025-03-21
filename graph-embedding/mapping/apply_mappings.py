import os
import json


def apply_mapping_to_predicate(line, predicate_mapping, type_mapping, is_modes_file=False):
    """
    Aplica o mapeamento a uma linha contendo um predicate. Se o predicate não tiver um mapeamento, retorna None.
    Se a linha não for um predicate válido, mantém a linha original.
    """
    line = line.strip()

    # Se for uma linha vazia ou um comentário, mantém a linha
    if not line or line.startswith("%"):
        return line, None

    # Extrair o predicado e os argumentos
    predicate = line.split("(")[0]
    args = line.split("(")[1][:-2].split(",")

    # Se o predicate não tiver um mapeamento, remover a linha
    if predicate not in predicate_mapping or predicate_mapping[predicate] is None:
        return None, None

    if is_modes_file:
        args = [arg[0] + type_mapping[arg[1:]] for arg in args]

    new_predicate = predicate_mapping[predicate]
    mapped_line = f"{new_predicate}({','.join(args)})."
    return mapped_line, new_predicate


def apply_mapping(source_path, output_path, predicate_mapping, type_mapping, target_predicates):
    """
    Processa o dataset aplicando os mapeamentos nos arquivos.

    - Se um predicado de facts.pl for mapeado para um predicado-alvo, ele vai para pos.pl
    - Se um predicado de pos.pl do domínio de origem **não for mapeado para um predicado-alvo**, ele vai para facts.pl
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Percorrer os folds
    for fold in sorted(os.listdir(source_path)):
        fold_path = os.path.join(source_path, fold)
        if not os.path.isdir(fold_path) or not fold.startswith("fold"):
            continue  # Ignorar arquivos que não sejam pastas "foldXX"

        # Criar diretório correspondente no output
        output_fold_path = os.path.join(output_path, fold)
        os.makedirs(output_fold_path, exist_ok=True)

        # Criar buffers para armazenar os conteúdos
        facts_content = []
        pos_content = []
        neg_content = []
        modes_content = set()

        # Processar cada arquivo dentro do fold
        for filename in ["facts.pl", "modes.pl", "pos.pl"]:
            file_path = os.path.join(fold_path, filename)
            if not os.path.exists(file_path):
                continue

            is_modes_file = filename == "modes.pl"  # Verifica se é o modes.pl
            with open(file_path, "r") as infile:
                for line in infile:
                    mapped_line, mapped_predicate = apply_mapping_to_predicate(line,
                                                                               predicate_mapping,
                                                                               type_mapping,
                                                                               is_modes_file)

                    if mapped_line:
                        if filename == "facts.pl":
                            # Se um fato foi mapeado para um predicado-alvo, mover para pos.pl
                            if mapped_predicate in target_predicates:
                                pos_content.append(mapped_line)
                            else:
                                facts_content.append(mapped_line)
                        elif filename == "pos.pl":
                            # Se um exemplo positivo não for mapeado para um predicado-alvo, mover para facts.pl
                            if mapped_predicate in target_predicates:
                                pos_content.append(mapped_line)
                            else:
                                facts_content.append(mapped_line)
                        elif filename == "modes.pl":
                            modes_content.add(mapped_line)

        # Salvar arquivos
        with open(os.path.join(output_fold_path, "facts.pl"), "w") as f:
            f.write("\n".join(facts_content) + "\n")
        with open(os.path.join(output_fold_path, "pos.pl"), "w") as f:
            f.write("\n".join(pos_content) + "\n")
        with open(os.path.join(output_fold_path, "modes.pl"), "w") as f:
            f.write("\n".join(modes_content) + "\n")


def apply_mappings(source_path, output_path, mappings, target_predicate):
    with open(os.path.join(output_path, "mappings.json"), 'w') as f:
        json.dump(mappings, f)

    for index, mapping in enumerate(mappings):
        index_path = os.path.join(output_path, str(index))
        os.makedirs(index_path, exist_ok=True)
        apply_mapping(source_path, index_path, mapping["predicate_mappings"], mapping["type_mappings"], target_predicate)
