import os
import re


class Mapping:
    def __init__(self, source_modes, target_modes):
        self._source_modes = source_modes
        self._source_types = self._extract_types(source_modes)
        self._source_types_embeddings = {}
        self._source_predicate_embeddings = {}

        self._target_modes = target_modes
        self._target_types = self._extract_types(target_modes)
        self._target_types_embeddings = {}
        self._target_predicate_embeddings = {}
        self._target_target_predicate = None
        self._target_target_types = None

        self._coherent_mappings = {}
        self._useful_mappings = {}
        self._robust_mappings = {}

    def _extract_types(self, modes):
        types = set()

        for value in modes.values():
            for item in value:
                types.add(item)

        return types

    def get_source_predicates(self):
        return list(self._source_modes.keys())

    def get_target_predicates(self):
        return list(self._target_modes.keys())

    def get_target_modes(self):
        return self._target_modes

    def set_coherent_mappings(self, mappings):
        self._coherent_mappings = mappings

    def set_useful_mappings(self, mappings):
        self._useful_mappings = mappings

    def get_target_predicate_arg_types(self, predicate):
        return self._target_modes[predicate][0]

    def get_useful_mappings(self):
        return self._useful_mappings

    def set_target_target_predicate(self, target_target_predicate):
        self._target_target_predicate = target_target_predicate
        self._target_target_types = set(self._target_modes[target_target_predicate])

    def get_target_target_predicate(self):
        return self._target_target_predicate

    def get_target_target_types(self):
        return self._target_target_types

    def print_coherent_mappings(self):
        print("Total de soluções coerentes encontradas:", len(self._coherent_mappings))
        for sol_idx, mapping in enumerate(self._coherent_mappings, start=1):
            print("\nSolução", sol_idx)
            print("Mapeamento de tipos:")
            for src_type, tgt_type in mapping["type_mappings"].items():
                print(f"  {src_type} -> {tgt_type}")
            print("Mapeamento de predicados:")
            for key, value in mapping["predicate_mappings"].items():
                print("  " + key + " -> " + str(value))
            print("-" * 40)

    def print_useful_mappings(self):
        print("Total de soluções úteis encontradas:", len(self._useful_mappings))
        for index, mapping in enumerate(self._useful_mappings, start=1):
            print("\nSolução " + str(index))
            print("Mapeamento de tipos:")
            for key, value in mapping["type_mappings"].items():
                print("  " + key + " -> " + str(value))
            print("Mapeamento de predicados:")
            for key, value in mapping["predicate_mappings"].items():
                print("  " + key + " -> " + str(value))
            print("-" * 40)

    def set_robust_mappings(self, robust_mappings):
        self._robust_mappings = robust_mappings

    def get_robust_mappings(self):
        return self._robust_mappings

##############################
# 1. Funções para parse dos arquivos modes.pl
##############################

def _parse_mode_line(line):
    """
    Faz o parse de uma linha do arquivo modes.pl.
    Espera linhas no formato:
        predicate(arg1, arg2, ...).
    Ignora linhas vazias ou comentários (que começam com '%').
    """
    line = line.strip().replace("+", "").replace("-", "")
    if not line or line.startswith('%'):
        return None
    m = re.match(r'^(\w+)\((.+)\)\.$', line)
    if not m:
        return None
    predicate = m.group(1)
    args_str = m.group(2)
    args = [arg.strip() for arg in args_str.split(',')]
    return predicate, args

def _parse_modes_file(filepath):
    """
    Lê o arquivo modes.pl e retorna um dicionário.
    Cada chave é o nome de um predicado e o valor é uma lista de assinaturas
    (cada assinatura é uma lista de argumentos, ex: ['+Professor', '+Aluno']).
    """
    modes = {}
    with open(filepath, 'r') as f:
        for line in f:
            parsed = _parse_mode_line(line)
            if parsed is not None:
                pred, args = parsed
                if pred not in modes:
                    modes[pred] = args
    return modes

def _extract_types(signature):
    """
    Dada uma assinatura (lista de argumentos como ['+Professor', '+Aluno']),
    retorna a lista dos tipos, removendo os sinais.
    Exemplo: ['Professor', 'Aluno']
    """
    return [arg[1:] for arg in signature]

##############################
# 2. Geração de candidatos de mapeamento (somente com assinaturas)
##############################

def _is_candidate_mapping_valid(src_signature, tgt_signature):
    """
    Dadas uma assinatura da origem e uma do destino, verifica se é possível
    mapear uma para a outra. As assinaturas devem ter o mesmo número de argumentos.
    Para cada posição, mapeia o tipo da origem para o tipo do destino.
    Se o mesmo tipo da origem aparecer duas vezes, os tipos correspondentes do destino
    devem ser iguais.
    Retorna o dicionário (local mapping) se for válido, ou None se não for.
    """
    if len(src_signature) != len(tgt_signature):
        return None
    local_map = {}
    for src_type, tgt_type in zip(src_signature, tgt_signature):
        if src_type in local_map:
            if local_map[src_type] != tgt_type:
                return None  # conflito: mesmo tipo da origem, destinos diferentes
        else:
            local_map[src_type] = tgt_type
    return local_map


def generate_tasks(source_modes, target_modes):
    """
    Para cada predicado da origem, gera tarefas com possíveis mapeamentos.
    O número de tarefas será igual ao número de predicados do domínio de origem.
    """
    tasks = []

    for src_pred, src_signature in source_modes.items():
        task = {
            'source_predicate': src_pred,
            'source_signature': src_signature,
            'candidates': []
        }

        for tgt_pred, tgt_signature in target_modes.items():
            local_map = _is_candidate_mapping_valid(src_signature, tgt_signature)
            if local_map is not None:
                candidate = {
                    'target_predicate': tgt_pred,
                    'target_signature': tgt_signature,
                    'local_map': local_map
                }
                task['candidates'].append(candidate)

        # Adiciona a opção de mapeamento para nulo (remoção)
        task['candidates'].append({
            'target_predicate': None,
            'target_signature': None,
            'local_map': {}
        })
        tasks.append(task)

    return tasks


##############################
# 3. Busca global com consistência do mapeamento de tipos (backtracking)
##############################

def combine_mapping(global_map, local_map):
    """
    Tenta combinar o mapeamento local (de uma tarefa) com o mapeamento global acumulado.
    Se já houver um mapeamento para um determinado tipo e este for diferente do novo,
    retorna None (conflito). Caso contrário, retorna o novo mapeamento combinado.
    """
    new_map = global_map.copy()
    for k, v in local_map.items():
        if k in new_map:
            if new_map[k] != v:
                return None  # conflito
        else:
            new_map[k] = v
    return new_map


def search_tasks(tasks, global_map, current_assignment, solutions, index=0):
    """
    Realiza a busca recursiva (backtracking) para atribuir, para cada tarefa (predicado da origem),
    um candidato de mapeamento que seja consistente com o mapeamento global de tipos.
    Quando todas as tarefas são processadas, a solução (lista de atribuições e o mapeamento global final)
    é adicionada à lista de soluções.
    """
    if index == len(tasks):
        solutions.append((current_assignment.copy(), global_map.copy()))
        return
    task = tasks[index]
    for candidate in task['candidates']:
        new_global = combine_mapping(global_map, candidate['local_map'])
        if new_global is not None:
            current_assignment.append({
                'source_predicate': task['source_predicate'],
                'source_signature': task['source_signature'],
                'target_predicate': candidate['target_predicate'],
                'target_signature': candidate['target_signature'],
                'local_map': candidate['local_map']
            })
            search_tasks(tasks, new_global, current_assignment, solutions, index + 1)
            current_assignment.pop()

def search_tasks_iterative(tasks):
    solutions = []
    stack = [(0, [], {})]  # Pilha que armazena o estado: (índice da tarefa, atribuição atual, mapeamento global)

    while stack:
        index, current_assignment, global_map = stack.pop()

        if index == len(tasks):
            # Caso base: todas as tarefas foram atribuídas
            solutions.append((current_assignment.copy(), global_map.copy()))
            continue

        task = tasks[index]
        for candidate in task['candidates']:
            # Tenta combinar o mapeamento global com o mapeamento local do candidato
            new_global = combine_mapping(global_map, candidate['local_map'])
            if new_global is not None:
                # Só adiciona a tarefa se a atribuição está sendo feita para um predicado de destino ainda não mapeado
                if not (candidate["target_predicate"] is not None and
                        candidate["target_predicate"] in[item["target_predicate"] for item in current_assignment]):
                    # Adiciona a atribuição atualizada e avança para a próxima tarefa
                    new_assignment = current_assignment + [{
                        'source_predicate': task['source_predicate'],
                        'source_signature': task['source_signature'],
                        'target_predicate': candidate['target_predicate'],
                        'target_signature': candidate['target_signature'],
                        'local_map': candidate['local_map']
                    }]
                    stack.append((index + 1, new_assignment, new_global))

    return solutions


##############################
# 4. Função principal
##############################

def generate_all_coherent_mappings(source_modes, target_modes):
    # Gera as tarefas (agrupando assinaturas idênticas, ignorando os sinais)
    tasks = generate_tasks(source_modes, target_modes)

    # Executa a busca global (backtracking)
    solutions = []
    #search_tasks(tasks, {}, [], solutions)
    solutions = search_tasks_iterative(tasks)

    out = []
    for solution in solutions:
        out.append(
            {
                "predicate_mappings": {item["source_predicate"]: item["target_predicate"] for item in solution[0]},
                "type_mappings": solution[1]
            }
        )

    return out

##############################
# 5. Filtra os mapeamentos que contém o predicado alvo e seus tipos
##############################

def filter_solutions(solutions, required_predicate):
    """
    Filtra as soluções de mapeamento, mantendo apenas aquelas que possuem
    mapeamento para todos os tipos obrigatórios e que utilizam todos os
    predicados obrigatórios.

    Cada solução é uma tupla (assignment, global_mapping), onde:
      - global_mapping: dicionário que mapeia os tipos da origem para os tipos do destino.
      - assignment: lista de dicionários, cada um com as chaves:
          'source_predicate', 'source_signature', 'target_predicate', 'target_signature', 'local_map'

    :param solutions: Lista de soluções geradas pelo código anterior.
    :param required_types: Lista de tipos obrigatórios (por exemplo, ['person', 'movie']).
    :param required_predicate: Predicado obrigatório (por exemplo, 'actor'.
    :return: Lista de soluções que atendem aos critérios.
    """

    return [mapping for mapping in solutions if required_predicate in mapping["predicate_mappings"].values()]

##############################
# 5. Filtra os mapeamentos que populam o arquivo facts.pl
##############################

def filter_robust_solutions(mappings):
    """
    Filtra as soluções de mapeamento, mantendo apenas aquelas que possuem
    mapeamento para todos os tipos obrigatórios e que utilizam todos os
    predicados obrigatórios.

    Cada solução é uma tupla (assignment, global_mapping), onde:
      - global_mapping: dicionário que mapeia os tipos da origem para os tipos do destino.
      - assignment: lista de dicionários, cada um com as chaves:
          'source_predicate', 'source_signature', 'target_predicate', 'target_signature', 'local_map'

    :param solutions: Lista de soluções geradas pelo código anterior.
    :param required_types: Lista de tipos obrigatórios (por exemplo, ['person', 'movie']).
    :param required_predicates: Lista de predicados obrigatórios (por exemplo, ['actor', 'director']).
    :return: Lista de soluções que atendem aos critérios.
    """
    filtered = []

    for mapping in mappings.get_useful_mappings():

        mapped_predicates = set(mapping["predicate_mappings"].values())
        if None in mapped_predicates:
            mapped_predicates.remove(None)

        # Considera apenas os mapeamentos que mapeiam predicados além do predicado target e que mapeiam predicados que utilizam os tipos dos argumentos do target
        if len(mapped_predicates) > 1:
            mapped_predicates.remove(mappings.get_target_target_predicate())

            types_used = []
            for key in mappings.get_target_modes():
                if key in mapped_predicates:
                    types_used += mappings.get_target_modes()[key]
            types_used = set(types_used)

            if any(target_type in types_used for target_type in mappings.get_target_target_types()):
                filtered.append(mapping)

    mappings.set_robust_mappings(filtered)

def get_required_predicate(filepath):
    predicate = ""
    with open(filepath, 'r') as f:
        for line in f:
            predicate = line.split("(")[0]
            break
    f.close()
    return predicate


def generate_mappings(source_dir, target_dir):
    # Caminhos para os arquivos modes.pl da origem e do destino
    source_modes_filepath = os.path.join(source_dir, 'modes.pl')
    target_modes_filepath = os.path.join(target_dir, 'modes.pl')

    #print("Lendo vocabulário de origem:", source_modes_filepath)
    source_modes = _parse_modes_file(source_modes_filepath)

    #print("Lendo vocabulário de destino:", target_modes_filepath)
    target_modes = _parse_modes_file(target_modes_filepath)

    mappings = Mapping(source_modes, target_modes)

    coherent_mappings = generate_all_coherent_mappings(source_modes, target_modes)

    mappings.set_coherent_mappings(coherent_mappings)

    target_pos_filepath = os.path.join(target_dir, 'pos.pl')
    required_predicate = get_required_predicate(target_pos_filepath)

    mappings.set_target_target_predicate(required_predicate)

    useful_mappings = filter_solutions(coherent_mappings, required_predicate)
    mappings.set_useful_mappings(useful_mappings)

    filter_robust_solutions(mappings)

    return mappings

