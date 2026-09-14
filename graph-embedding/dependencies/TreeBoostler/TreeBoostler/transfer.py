'''
   Functions to handle transfer learning of boosted trees
   Name:         transfer.py
   Author:       Rodrigo Azevedo
   Updated:      August 31, 2018
   License:      GPLv3
'''
#from revision import *
import copy
import re

#transfer_map = ['workedunder(A, B) -> advisedby(B, A)',
#            'director(A) -> professor(A)',
#            'actor(A) -> student(A)',
#            'movie(A, B) -> publication(B, B)'
#            ]
#
#structured = [['workedunder(A, B)',
#  {'': 'director(B), movie(C, A), movie(C, B)',
#   'false': 'teste(A,B)',
#   'false,false': 'movie(B,A)'},
#  {'false,true': [6.83e-08, 77, 0], 'false,false,false': [6.83e-08, 77, 0], 'false,false,true': [6.83e-08, 77, 0], 'true': [0.0, 0, 77]}],
# ['workedunder(A, B)',
#  {'': 'director(B), movie(C, A), movie(C, B)'},
#  {'false': [0.0, 77, 0], 'true': [2.23e-07, 0, 77]}],
# ['workedunder(A, B)',
#  {'': 'director(B), movie(C, A), movie(C, B)'},
#  {'false': [5.67e-08, 77, 0], 'true': [3.26e-07, 0, 77]}],
# ['workedunder(A, B)',
#  {'': 'director(B), movie(C, A), movie(C, B)'},
#  {'false': [5.27e-08, 77, 0], 'true': [0.0, 0, 77]}],
# ['workedunder(A, B)',
#  {'': 'director(B), movie(C, A), movie(C, B)'},
#  {'false': [0.0, 77, 0], 'true': [0.0, 0, 77]}],
# ['workedunder(A, B)',
#  {'': 'director(B), movie(C, A), movie(C, B)'},
#  {'false': [1.83e-08, 77, 0], 'true': [0.0, 0, 77]}],
# ['workedunder(A, B)',
#  {'': 'director(B), movie(C, A), movie(C, B)'},
#  {'false': [3.57e-08, 77, 0], 'true': [0.0, 0, 77]}],
# ['workedunder(A, B)',
#  {'': 'director(B), movie(C, A), movie(C, B)'},
#  {'false': [0.0, 77, 0], 'true': [2.98e-08, 0, 77]}],
# ['workedunder(A, B)',
#  {'': 'director(B), movie(C, A), movie(C, B)'},
#  {'false': [2.89e-08, 77, 0], 'true': [7.88e-08, 0, 77]}],
# ['workedunder(A, B)',
#  {'': 'director(B), movie(C, A), movie(C, B)'},
#  {'false': [0.0, 77, 0], 'true': [5.37e-08, 0, 77]}]]

class transfer:
    def literal_to_str(literal):
        '''Generate literal string from tuple'''
        return literal[0] + '(' + literal[1] + ')'

    def transfer_variables(variables, from_map, to_map):
        '''Transfer variables of a clause given a mapping'''
        read = {}
        ret = []
        values = variables.split(',')
        f = from_map.split(',')
        t = to_map.split(',')
        for i in range(len(values)):
            read[f[i].strip()] = values[i].strip()
        for i in range(len(t)):
            ret.append(read[t[i].strip()])
        return ', '.join(ret)

    def get_mapping_struct(mapping):
        '''Generate mapping structure'''
        ret = {}
        for m in mapping:
            match = re.findall('([a-zA-Z_0-9]*)\s*\(([a-zA-Z_0-9,\s]*)\)', m)
            if match:
                ret[match[0][0]] = (match[1][0], match[0][1], match[1][1])
        return ret

    def transfer_literal(literal, mapping_struct):
        if literal[0] in mapping_struct:
            m = mapping_struct[literal[0]]
            return (m[0], transfer.transfer_variables(literal[1], m[1], m[2]))
        else:
            return None

    def get_transfer_tree_helper(path, nodes, leaves):
        children = [None, None]
        split = [] if path == '' else path.split(',')
        left = ','.join(split+['true'])
        right = ','.join(split+['false'])
        if left in nodes:
            children[0] = transfer.get_transfer_tree_helper(left, nodes, leaves)
        if right in nodes:
            children[1] = transfer.get_transfer_tree_helper(right, nodes, leaves)
        if left in leaves:
            children[0] = leaves[left] # { 'type': 'leaf', 'std_dev': leaves[left][0], 'neg': leaves[left][1], 'pos': leaves[left][2] }
        if right in leaves:
            children[1] = leaves[right]
        return { nodes[path]: children }
        # { 'type': 'node', 'literals': nodes[path], 'children': children, 'variavarc] }

    def get_transfer_tree(nodes, leaves):
        return transfer.get_transfer_tree_helper('', nodes, leaves)

    def add_subtree_to_false(root, subtree):
        if isinstance(root, list):
            return root
        elif isinstance(root, dict):
            node_str = list(root.keys())[0]
            value = root[node_str]
            children = value
            true_child = children[0]
            false_child = transfer.add_subtree_to_false(children[1], subtree)
            if isinstance(false_child, list):
                return { node_str: [true_child, subtree] }
            else:
                return { node_str: [true_child, false_child] }

    def merge_subtrees(left, right):
        # if TRUE node is leaf then return FALSE node
        if isinstance(left, list):
            return right
        # otherwise
        left_str = list(left.keys())[0]
        value = left[left_str]
        children = value
        true_child = children[0]
        false_child = children[1]
        new_true_child = transfer.add_subtree_to_false(true_child, false_child)
        return { left_str: [new_true_child, right] }

    def transfer_tree_helper(root, mapping_struct):
        if isinstance(root, list):
            return root
        elif isinstance(root, dict):
            node_str = list(root.keys())[0]
            value = root[node_str]
            children = value
            true_child = transfer.transfer_tree_helper(children[0], mapping_struct)
            false_child = transfer.transfer_tree_helper(children[1], mapping_struct)
            match = re.findall('([a-zA-Z_0-9]*)\s*\(([a-zA-Z_0-9,\s]*)\)', node_str)
            new_clause = []
            if match:
                for m in match:
                    transfered = transfer.transfer_literal(m, mapping_struct)
                    if transfered:
                        new_clause.append(transfer.literal_to_str(transfered))
            if len(new_clause):
                new_key = ', '.join(new_clause)
                return { new_key: [true_child, false_child] }
            else:
                # nodes with no literals should be replaced by its false subtree
                #return false_child
                # nodes with no literals are replaced by its true child and merge with false child
                return transfer.merge_subtrees(true_child, false_child)

    def transfer_tree(tree, mapping_struct):
        ntree = copy.deepcopy(tree)
        return transfer.transfer_tree_helper(ntree, mapping_struct)

    def get_structured_from_transfer_tree_helper(path, root, nodes, leaves):
        if isinstance(root, list):
            leaves[path] = root
        elif isinstance(root, dict):
            i = list(root.keys())[0]
            value = root[i]
            children = value
            split = [] if path == '' else path.split(',')
            left = ','.join(split+['true'])
            right = ','.join(split+['false'])
            nodes[path] = i
            transfer.get_structured_from_transfer_tree_helper(left, children[0], nodes, leaves)
            transfer.get_structured_from_transfer_tree_helper(right, children[1], nodes, leaves)

    def get_structured_from_transfer_tree(target, tree):
        nodes = {}
        leaves = {}
        transfer.get_structured_from_transfer_tree_helper('', tree, nodes, leaves)
        return [target, nodes, leaves]

    def transfer(structured, mapping):
        '''Transfer structure according to mapping'''
        copied = copy.deepcopy(structured)
        mapping_struct = transfer.get_mapping_struct(mapping)
        for struct in copied:
            match = re.match('([a-zA-Z_0-9]*)\s*\(([a-zA-Z_0-9,\s]*)\)', struct[0])
            if match:
                transfered = transfer.transfer_literal(match.groups(), mapping_struct)
                if transfered:
                    struct[0] = transfer.literal_to_str(transfered)
                else:
                    raise(Exception('Attempted to transfer head to a None mapping.'))
            else:
                 raise(Exception('Attempted to transfer head that does not exist.'))
            # nodes with no literals should be replaced by its subtree
            tree = transfer.get_transfer_tree(struct[1], struct[2])
            transferred = transfer.transfer_tree(tree, mapping_struct)
            new_struct = transfer.get_structured_from_transfer_tree(struct[0], transferred)
    #        remove_nodes = []
    #        new_nodes = {}
    #        for key, value in struct[1].items():
    #            if ','.join((key.split(','))[:-1]) not in remove_nodes:
    #                match = re.findall('([a-zA-Z_0-9]*)\s*\(([a-zA-Z_0-9,\s]*)\)', value)
    #                new_clause = []
    #                if match:
    #                    for m in match:
    #                        transfered = transfer_literal(m, mapping_struct)
    #                        if transfered:
    #                            new_clause.append(literal_to_str(transfered))
    #                if len(new_clause):
    #                    #struct[1][key] = ', '.join(new_clause)
    #                    new_nodes[key] = ', '.join(new_clause)
    #                else:
    #                    remove_nodes.append(key)
            struct[1] = new_struct[1]
            struct[2] = new_struct[2]
        return copied

    def get_transferred_target(structured):
        '''Remove target from structured tree'''
        target = structured[0][0]
        match = re.match('([a-zA-Z_0-9]*)\s*\(([a-zA-Z_0-9,\s]*)\)', target)
        if match:
            return match.groups()[0]

    def get_transfer_file(source_bk, target_bk, from_pred, to_pred, recursion=False, searchArgPermutation=False, searchEmpty=False, allowSameTargetMap=False):
        srcSet = set()
        from_arity = 0
        to_arity = 0
        for pred in source_bk:
            a = re.sub('[\+\-\`]', '', pred)
            srcSet.add(a)
            match = re.match('([a-zA-Z_0-9]*)\s*\(([a-zA-Z_0-9,\s]*)\)', a)
            if match:
                g = match.groups()
                if g[0] == from_pred:
                    from_arity = len(g[1].split(','))
        tarSet = set()
        for pred in target_bk:
            a = re.sub('[\+\-\`]', '', pred)
            tarSet.add(a)
            match = re.match('([a-zA-Z_0-9]*)\s*\(([a-zA-Z_0-9,\s]*)\)', a)
            if match:
                g = match.groups()
                if g[0] == to_pred:
                    to_arity = len(g[1].split(','))
        tra = []
        for item in srcSet:
            tra.append('source: ' + item)
        for item in tarSet:
            tra.append('target: ' + item)
        tra.append('setMap: ' + from_pred + '(' + ','.join([chr(65+i) for i in range(from_arity)]) + ')=' + to_pred + '(' + ','.join([chr(65+i) for i in range(to_arity)]) + ').')
        if recursion:
            tra.append('setMap: recursion_' + from_pred + '(A,B)=recursion_' + to_pred + '(A,B).')
        tra.append('setParam: searchArgPermutation=' + str(searchArgPermutation).lower() + '.')
        tra.append('setParam: searchEmpty=' + str(searchEmpty).lower() + '.')
        tra.append('setParam: allowSameTargetMap=' + str(allowSameTargetMap).lower() + '.')
        return tra
