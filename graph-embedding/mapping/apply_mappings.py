import os


def apply_mapping_to_predicate(line, predicate_mapping, type_mapping, is_modes_file=False):
    """
    Aplica o mapeamento a uma linha contendo um predicate. Se o predicate não tiver um mapeamento, retorna None.
    Se a linha não for um predicate válido, mantém a linha original.
    """

    line = line.strip()

    # Se for uma linha vazia ou um comentário, mantém a linha
    if not line or line.startswith("%"):
        return line

    # Verificar se a linha corresponde a um predicate na forma predicate(arg1, arg2).
    predicate = line.split("(")[0]
    args = line.split("(")[1][:-2].split(",")

    # Se o predicate não tiver um mapeamento, remover a linha
    if predicate not in predicate_mapping or predicate_mapping[predicate] is None:
        return None

    if is_modes_file:
        args = [arg[0] + type_mapping[arg[1:]] for arg in args]

    new_predicate = predicate_mapping[predicate]
    mapped_line = f"{new_predicate}({','.join(args)})."
    return mapped_line


def apply_mapping(source_path, output_path, predicate_mapping, type_mapping):
    """
    Processa o dataset aplicando os mapeamentos nos arquivos.

    :param source_path: Caminho do dataset original.
    :param output_path: Caminho onde o dataset transformado será salvo.
    :param predicate_mapping: Dicionário de mapeamento de predicates.
    :param type_mapping: Dicionário de mapeamento de tipos.
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

        # Processar cada arquivo dentro do fold
        for filename in ["facts.pl", "modes.pl", "neg.pl", "pos.pl"]:
            file_path = os.path.join(fold_path, filename)
            if not os.path.exists(file_path):
                continue

            output_file_path = os.path.join(output_fold_path, filename)
            is_modes_file = filename == "modes.pl"  # Verifica se é o modes.pl

            with open(file_path, "r") as infile, open(output_file_path, "w") as outfile:

                for line in infile:
                    mapped_line = apply_mapping_to_predicate(line, predicate_mapping, type_mapping, is_modes_file)
                    if mapped_line:
                        outfile.write(mapped_line + "\n")

def apply_mappings(source_path, output_path, mappings):
    import json

    with open(output_path + "/mappings.json", 'w') as f:
        json.dump(mappings, f)

    for index, mapping in enumerate(mappings):
        if not os.path.exists(output_path + "/" + str(index)):
            os.makedirs(output_path + "/" + str(index))
        apply_mapping(source_path, output_path + "/" + str(index), mapping["predicate_mappings"], mapping["type_mappings"])

