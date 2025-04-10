import os
import json
import itertools
import random

from torch.cuda import graph

from mapping.mapping import parse_modes_file


def apply_mapping_to_predicate(line, predicate_mapping):
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

    new_predicate = predicate_mapping[predicate]
    mapped_line = f"{new_predicate}({','.join(args)})."
    return mapped_line, new_predicate

def generate_neg(facts, pos, target_modes):
    target_predicate = pos[0].split("(")[0]

    type1 = target_modes[target_predicate][0]
    type2 = target_modes[target_predicate][1]

    type1_instances_set = set()
    type1_instances_list = []
    type2_instances_set = set()
    type2_instances_list = []

    for item in facts + pos:
        predicate = item.split("(")[0]
        args_types = target_modes[predicate]
        args = item.split("(")[1][:-2].split(",")

        for index, arg_type in enumerate(args_types):
            if arg_type == type1:
                type1_instances_set.add(args[index])
                type1_instances_list.append(args[index])
            if arg_type == type2:
                type2_instances_set.add(args[index])
                type2_instances_list.append(args[index])

    neg = []

    while len(neg) < len(pos) * 2:
        sample_index1 = random.randrange(len(type1_instances_list))
        sample_index2 = random.randrange(len(type2_instances_list))
        sample = f"{target_predicate}({type1_instances_list[sample_index1]},{type2_instances_list[sample_index2]})."
        if sample not in pos and sample not in neg:
            neg.append(sample)

    return neg

def apply_mapping(source_path, output_path, predicate_mapping, target_predicate, target_modes):
    """
    Processa o dataset aplicando os mapeamentos nos arquivos.

    - Se um predicado de facts.pl for mapeado para um predicado-alvo, ele vai para pos.pl
    - Se um predicado de pos.pl do domínio de origem **não for mapeado para um predicado-alvo**, ele vai para facts.pl
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Criar buffers para armazenar os conteúdos
    facts_content = []
    pos_content = []

    # Percorrer os folds
    for fold in sorted(os.listdir(source_path)):
        fold_path = os.path.join(source_path, fold)
        if not os.path.isdir(fold_path) or not fold.startswith("fold"):
            continue  # Ignorar arquivos que não sejam pastas "foldXX"

        facts = []
        pos = []

        # Processar cada arquivo dentro do fold
        for filename in ["facts.pl", "pos.pl"]:
            file_path = os.path.join(fold_path, filename)
            if not os.path.exists(file_path):
                continue

            with open(file_path, "r") as infile:
                for line in infile:
                    mapped_line, mapped_predicate = apply_mapping_to_predicate(line, predicate_mapping)

                    if mapped_line:
                        if filename == "facts.pl":
                            # Se um fato foi mapeado para um predicado-alvo, mover para pos.pl
                            if mapped_predicate == target_predicate:
                                pos.append(mapped_line)
                            else:
                                facts.append(mapped_line)
                        elif filename == "pos.pl":
                            # Se um exemplo positivo não for mapeado para um predicado-alvo, mover para facts.pl
                            if mapped_predicate == target_predicate:
                                pos.append(mapped_line)
                            else:
                                facts.append(mapped_line)

        # Gerar neg.pl aqui
        if facts and pos:
            facts_content.append(facts)
            pos_content.append(pos)

    return facts_content, pos_content

def save_mapping(output_path, facts, pos, neg):
    with open(os.path.join(output_path, "facts.pl"), "w") as f:
        f.write("\n".join(facts) + "\n")
    with open(os.path.join(output_path, "pos.pl"), "w") as f:
        f.write("\n".join(pos) + "\n")
    with open(os.path.join(output_path, "neg.pl"), "w") as f:
        f.write("\n".join(neg) + "\n")


def apply_mappings(source_path, output_path, mappings, target_predicate, target_modes):
    with open(os.path.join(output_path, "mappings.json"), 'w') as f:
        json.dump(mappings, f)

    for index, mapping in enumerate(mappings):
        index_path = os.path.join(output_path, str(index))
        os.makedirs(index_path, exist_ok=True)
        apply_mapping(source_path, index_path, mapping["predicate_mappings"], target_predicate, target_modes)

def read_domain(domain_path, folds = None):
    """
    Processa o dataset aplicando os mapeamentos nos arquivos.

    - Se um predicado de facts.pl for mapeado para um predicado-alvo, ele vai para pos.pl
    - Se um predicado de pos.pl do domínio de origem **não for mapeado para um predicado-alvo**, ele vai para facts.pl
    """

    modes = parse_modes_file(os.path.join(domain_path, "fold00", "modes.pl"))

    # Criar buffers para armazenar os conteúdos
    domain_data = {
        "nodes": {},
        "edges": {}
    }

    for predicate, arg_types in modes.items():
        if predicate not in domain_data["edges"]:
            domain_data["edges"][predicate] = set()

            for arg_type in arg_types:
                if arg_type not in domain_data["nodes"]:
                    domain_data["nodes"][arg_type] = set()

    folds = folds if folds is not None else sorted(os.listdir(domain_path))

    # Percorrer os folds
    for fold in folds:
        fold_path = os.path.join(domain_path, fold)
        if not os.path.isdir(fold_path) or not fold.startswith("fold"):
            continue  # Ignorar arquivos que não sejam pastas "foldXX"

        # Processar cada arquivo dentro do fold
        for filename in ["facts.pl", "pos.pl"]:
            file_path = os.path.join(fold_path, filename)
            if not os.path.exists(file_path):
                continue

            with open(file_path, "r") as infile:
                for line in infile:
                    line = line.strip()
                    predicate = line.split("(")[0]
                    args = line.split("(")[1][:-2].split(",")
                    domain_data["edges"][predicate].add(tuple(args))
                    for index, arg in enumerate(args):
                        arg_type = modes[predicate][index]
                        domain_data["nodes"][arg_type].add(arg)

    return domain_data
