'''
   Functions to handle revision theory of boosted trees
   Name:         revision.py
   Author:       Rodrigo Azevedo
   Updated:      July 22, 2018
   License:      GPLv3
'''

import shutil
import os
import re
import copy
import math

class revision:

    @staticmethod
    def delete_train_files():
        '''Remove files from train folder'''
        try:
            shutil.rmtree('tboostsrl/train')
        except:
            pass
        try:
            os.remove('tboostsrl/train_output.txt')
        except:
            pass

    @staticmethod
    def delete_test_files():
        '''Remove files from test folder'''
        try:
            shutil.rmtree('tboostsrl/test')
        except:
            pass
        try:
            os.remove('tboostsrl/test_output.txt')
        except:
            pass

    @staticmethod
    def delete_model_files():
        '''Remove files of last model'''
        revision.delete_train_files()
        revision.delete_test_files()

    @staticmethod
    def save_model_files():
        '''Remove files of last model as best model'''
        try:
            shutil.rmtree('tboostsrl/best')
        except:
            pass
        os.mkdir('tboostsrl/best')
        shutil.move('tboostsrl/train', 'tboostsrl/best')
        shutil.move('tboostsrl/test', 'tboostsrl/best')
        shutil.move('tboostsrl/train_output.txt', 'tboostsrl/best')
        shutil.move('tboostsrl/test_output.txt', 'tboostsrl/best')

    @staticmethod
    def get_saved_model_files():
        '''Recover model files of best model'''
        shutil.move('tboostsrl/best/train', 'tboostsrl')
        shutil.move('tboostsrl/best/test', 'tboostsrl')
        shutil.move('tboostsrl/best/train_output.txt', 'tboostsrl')
        shutil.move('tboostsrl/best/test_output.txt', 'tboostsrl')
        try:
            shutil.rmtree('tboostsrl/best')
        except:
            pass

    @staticmethod
    def get_tree_helper(path, nodes, leaves, variances, no_variances=False):
        children = [None, None]
        split = [] if path == '' else path.split(',')
        left = ','.join(split+['true'])
        right = ','.join(split+['false'])
        varc = variances[path] if not no_variances else []
        if left in nodes:
            children[0] = revision.get_tree_helper(left, nodes, leaves, variances, no_variances=no_variances)
        if right in nodes:
            children[1] = revision.get_tree_helper(right, nodes, leaves, variances, no_variances=no_variances)
        if left in leaves:
            children[0] = leaves[left] # { 'type': 'leaf', 'std_dev': leaves[left][0], 'neg': leaves[left][1], 'pos': leaves[left][2] }
        if right in leaves:
            children[1] = leaves[right]
        return { nodes[path]: [varc, children] }
        # { 'type': 'node', 'literals': nodes[path], 'children': children, 'variavarc] }

    @staticmethod
    def get_tree(nodes, leaves, variances, no_variances=False):
        return revision.get_tree_helper('', nodes, leaves, variances, no_variances=no_variances)

    @staticmethod
    def generalize_tree_helper(root):
        if isinstance(root, list):
            return root
        elif isinstance(root, dict):
            i = list(root.keys())[0]
            value = root[i]
            children= value[1]
            variances = value[0]
            true_child = revision.generalize_tree_helper(children[0])
            false_child = revision.generalize_tree_helper(children[1])
            # if TRUE child has 0 examples reached
            if math.isnan(variances[0]):
                return false_child
            # if FALSE child has 0 examples reached
            if math.isnan(variances[1]):
                return true_child
            # if node has only leaves
            if isinstance(true_child, list) and isinstance(false_child, list):
                if variances[0] >= 0.0025 and variances[1] >= 0.0025:
                    return [0, true_child[1] + false_child[1], true_child[2] + false_child[2]] # return a leaf
            # otherwise
            return { i: [variances, [true_child, false_child]] }

    @staticmethod
    def generalize_tree(tree):
        ntree = copy.deepcopy(tree)
        return revision.generalize_tree_helper(ntree)

    @staticmethod
    def get_structured_from_tree_helper(path, root, nodes, leaves):
        if isinstance(root, list):
            leaves[path] = root
        elif isinstance(root, dict):
            i = list(root.keys())[0]
            value = root[i]
            children= value[1]
            split = [] if path == '' else path.split(',')
            left = ','.join(split+['true'])
            right = ','.join(split+['false'])
            nodes[path] = i
            revision.get_structured_from_tree_helper(left, children[0], nodes, leaves)
            revision.get_structured_from_tree_helper(right, children[1], nodes, leaves)

    @staticmethod
    def get_structured_from_tree(target, tree):
        nodes = {}
        leaves = {}
        revision.get_structured_from_tree_helper('', tree, nodes, leaves)
        return [target, nodes, leaves]

    @staticmethod
    def print_will_produced_tree(will):
        '''Remove files from train folder'''
        for w in will:
            print(w)

    @staticmethod
    def descendant_of(path, leaves):
        if len(leaves) == 0:
            return False
        if '' in leaves:
            return True
        split = path.split(',')
        paths = set([','.join(split[:i+1]) for i in range(len(split))])
        intsc = paths.intersection(set(leaves))
        return True if len(intsc) else False

    @staticmethod
    def get_clause(struct, path):
        '''Get @staticmethod
    definite clause of given path'''
        target = struct[0]
        nodes = struct[1]
        paths = path.split(',')
        clauses = []
        for i in range(len(paths)):
            p = ','.join(paths[:i])
            t = paths[i]
            if t == 'true':
                clauses.append(nodes[p])
        return target + ' :- ' + ', '.join(clauses) + '.'

    @staticmethod
    def is_bad_leaf(value):
        '''@Staticmethod
    Defines if given leaf is bad or not (revision point).
        If leaf has 0 pos and 0 neg examples keeps it. What should be done?'''
        if sum(value[1:]) == 0:
            return True
        return max(value[1:])/sum(value[1:]) < 1.0

    @staticmethod
    def get_bad_leaf_value(value):
        '''
        Defines if given leaf is bad or not (revision point).
        If leaf has 0 pos and 0 neg examples keeps it. What should be done?'''
        if sum(value[1:]) == 0:
            return 0
        return max(value[1:])/sum(value[1:])

    @staticmethod
    def get_bad_leaves(struct):
        '''Get revision points (bad leaves)'''
        leaves = struct[2]
        bad_leaves = {}
        for path, value in leaves.items():
            # if it is a bad leaf, add it to bad_leaves
            if revision.is_bad_leaf(value):
                bad_leaves[path] = revision.get_bad_leaf_value(value)
        ret = [(path, value) for path, value in bad_leaves.items()]
        ret.sort(key=lambda x: x[1])
        return ret

    @staticmethod
    def get_candidate(struct, variances, treenumber=1, no_pruning=False):
        '''Get candidate refining every revision point in a tree'''
        target = struct[0]
        nodes = struct[1]
        leaves = struct[2]
        if '' not in nodes:
            return []
        tree = revision.get_tree(nodes, leaves, variances)
        gen = revision.generalize_tree(tree) if not no_pruning else tree
        new_struct = revision.get_structured_from_tree(target, gen)
        return revision.get_refine_file(new_struct, forceLearning=True, treenumber=treenumber)

    @staticmethod
    def get_boosted_candidate(structs, variances, no_pruning=False):
        refine = []
        for i in range(len(structs)):
            refine += revision.get_candidate(structs[i], variances[i], i+1, no_pruning=no_pruning)
        return refine

    @staticmethod
    def get_branch_with(branch, next_branch):
        '''Append next_branch at branch'''
        if not branch:
            return next_branch
        b = branch.split(',')
        b.append(next_branch)
        return ','.join(b)

    @staticmethod
    def get_branch_last_level(branch, new_branch):
        '''Returns a branch where last level has new path'''
        b = branch.split(',')
        b[-1] = new_branch
        return ','.join(b)

    @staticmethod
    def get_branch_to_last_level(branch):
        '''Returns a branch without last level'''
        b = branch.split(',')
        return ','.join(b[:-1])

    @staticmethod
    def get_refine_file(struct, forceLearning=False, treenumber=1):
        '''Generate the refine file from given tree structure'''
        target = struct[0]
        nodes = struct[1]
        #leaves = struct[2]
        tree = treenumber-1
        refine = []
        for path, value in nodes.items():
            node = target + ' :- ' + value + '.' if not path else value + '.'
            branchTrue = 'true' if revision.get_branch_with(path, 'true') in nodes or forceLearning else 'false'
            branchFalse = 'true' if revision.get_branch_with(path, 'false') in nodes or forceLearning else 'false'
            refine.append(';'.join([str(tree), path, node, branchTrue, branchFalse]))
        return refine

    @staticmethod
    def get_boosted_refine_file(structs, forceLearning=False):
        refine = []
        for i in range(len(structs)):
            refine += revision.get_refine_file(structs[i], treenumber=i+1, forceLearning=forceLearning)
        return refine

    @staticmethod
    def learn_model(background, tboostsrl, target, train_pos, train_neg, facts, refine=None, trees=10, print_function=None):
        '''Train and test a boosted or single tree'''
        revision.delete_model_files()
        model = tboostsrl.train(background, train_pos, train_neg, facts, refine=refine, trees=trees)
        will = ['WILL Produced-Tree #'+str(i+1)+'\n'+('\n'.join(model.get_will_produced_tree(treenumber=i+1))) for i in range(trees)]
        variances = [model.get_variances(treenumber=i+1) for i in range(trees)]
        if print_function:
            for i in will:
                print_function(i)
            print_function('\n')
        learning_time = model.traintime()
        structured = []
        for i in range(trees):
            structured.append(model.get_structured_tree(treenumber=i+1).copy())
        return [model, learning_time, structured, will, variances]

    @staticmethod
    def learn_test_model(background, tboostsrl, target, train_pos, train_neg, train_facts, test_pos, test_neg, test_facts, refine=None, transfer=None, trees=10, print_function=None):
        '''Train and test a boosted or single tree'''
        revision.delete_model_files()
        model = tboostsrl.train(background, train_pos, train_neg, train_facts, refine=refine, transfer=transfer, trees=trees)
        will = ['WILL Produced-Tree #'+str(i+1)+'\n'+('\n'.join(model.get_will_produced_tree(treenumber=i+1))) for i in range(trees)]
        variances = [model.get_variances(treenumber=i+1) for i in range(trees)]
        if print_function:
            for i in will:
                print_function(i)
            print_function('\n')
        learning_time = model.traintime()
        structured = []
        for i in range(trees):
            structured.append(model.get_structured_tree(treenumber=i+1).copy())
        results = tboostsrl.test(model, test_pos, test_neg, test_facts, trees=trees)
        inference_time = results.testtime()
        t_results = results.summarize_results()
        t_results['Learning time'] = learning_time
        t_results['Inference time'] = inference_time
        if print_function:
            print_function('Results')
            print_function('   AUC ROC   = %s' % t_results['AUC ROC'])
            print_function('   AUC PR    = %s' % t_results['AUC PR'])
            print_function('   CLL	      = %s' % t_results['CLL'])
            print_function('   Precision = %s at threshold = %s' % (t_results['Precision'][0], t_results['Precision'][1]))
            print_function('   Recall    = %s' % t_results['Recall'])
            print_function('   F1        = %s' % t_results['F1'])
            print_function('\n')
            print_function('Total learning time: %s seconds' % learning_time)
            print_function('Total inference time: %s seconds' % inference_time)
            print_function('AUC ROC: %s' % t_results['AUC ROC'])
            print_function('\n')
        return [model, t_results, structured, will, variances]

    @staticmethod
    def score_model(model, tboostsrl, test_pos, test_neg, test_facts, trees=10, print_function=None):
        results = tboostsrl.test(model, test_pos, test_neg, test_facts, trees=trees)
        inference_time = results.testtime()
        t_results = results.summarize_results()
        t_results['Inference time'] = inference_time
        if print_function:
            print_function('Results scoring model')
            print_function('   AUC ROC   = %s' % t_results['AUC ROC'])
            print_function('   AUC PR    = %s' % t_results['AUC PR'])
            print_function('   CLL	      = %s' % t_results['CLL'])
            print_function('   Precision = %s at threshold = %s' % (t_results['Precision'][0], t_results['Precision'][1]))
            print_function('   Recall    = %s' % t_results['Recall'])
            print_function('   F1        = %s' % t_results['F1'])
            print_function('\n')
            print_function('Total scoring time: %s seconds' % inference_time)
        return t_results

    @staticmethod
    def theory_revision(background, tboostsrl, target, r_train_pos, r_train_neg, train_facts, test_pos, test_neg, test_facts, structured_tree, trees=10, max_revision_iterations=1, transfer=None, print_function=None):
        '''Function responsible for starting the theory revision process'''
        total_revision_time = 0
        best_cll = - float('inf')
        best_structured = None
        best_model_results = None
        pl_t_results = 0

        # parameter learning
        if print_function:
            print_function('******************************************')
            print_function('Performing Parameter Learning')
            print_function('******************************************')
            print_function('Refine')
            for item in revision.get_boosted_refine_file(structured_tree):
                print_function(item)
            print_function('\n')
        [model, t_results, structured, will, variances] = revision.learn_test_model(background, tboostsrl, target, r_train_pos, r_train_neg, train_facts, test_pos, test_neg, test_facts, refine=revision.get_boosted_refine_file(structured_tree), transfer=transfer, trees=trees, print_function=print_function)
        # saving performed parameter learning will
        #tboostsrl.write_to_file(will, 'tboostsrl/last_will.txt')
        #tboostsrl.write_to_file([str(structured)], 'tboostsrl/last_structured.txt')
        pl_t_results = copy.deepcopy(t_results)

        # scoring model
        scored_results = revision.score_model(model, tboostsrl, r_train_pos, r_train_neg, train_facts, trees=trees, print_function=print_function)
        best_cll = scored_results['CLL']
        best_model_results = copy.deepcopy(t_results)
        total_revision_time = pl_t_results['Learning time'] + scored_results['Inference time']
        if print_function:
            print_function('Parameter learned model CLL: %s' % scored_results['CLL'])
            print_function('\n' )

        best_structured = copy.deepcopy(structured)
        if print_function:
            print_function('Structure after Parameter Learning')
            print_function(best_structured)
            print_function(variances)
            print_function('\n')
        revision.save_model_files()

        if print_function:
            print_function('******************************************')
            print_function('Performing Theory Revision')
            print_function('******************************************')
        # refine candidates
        for i in range(max_revision_iterations):
            if print_function:
                print_function('Refining iteration %s' % str(i+1))
                print_function('********************************')
            found_better = False
            candidate = revision.get_boosted_candidate(best_structured, variances)
            if not len(candidate):
                #return [model, copy.deepcopy(t_results), structured, pl_t_results]
                # Perform revision without pruning
                print_function('Pruning resulted in null theory\n')
                candidate = revision.get_boosted_candidate(best_structured, variances, no_pruning=True)
            if print_function:
                print_function('Candidate for revision')
                for item in candidate:
                    print_function(item)
                print_function('\n')
            #tboostsrl.write_to_file(candidate, 'tboostsrl/last_candidate.txt')
            if print_function:
                print_function('Refining candidate')
                print_function('***************************')
                #print('Revision points found')
                #for i in range(trees):
                #    print('Tree #%s: %s' % (i+1, str(get_bad_leaves(best_structured[i]))))
                #print('\n')
            [model, t_results, structured, will, variances] = revision.learn_test_model(background, tboostsrl, target, r_train_pos, r_train_neg, train_facts, test_pos, test_neg, test_facts, trees=trees, refine=candidate, print_function=print_function)
            #t_results['Learning time'] = t_results['Learning time'] + pl_t_results['Learning time']
            # scoring model
            scored_results = revision.score_model(model, tboostsrl, r_train_pos, r_train_neg, train_facts, trees=trees, print_function=print_function)
            total_revision_time = total_revision_time + t_results['Learning time'] + scored_results['Inference time']
            if scored_results['CLL'] > best_cll:
                found_better = True
                best_cll = scored_results['CLL']
                best_structured = copy.deepcopy(structured)
                best_model_results = copy.deepcopy(t_results)
                revision.save_model_files()
            if print_function:
                print_function('Refined model CLL: %s' % scored_results['CLL'])
                print_function('\n')
            if found_better == False:
                break

        # set total revision time to t_results learning time
        best_model_results['Learning time'] = total_revision_time
        # test best model
        if print_function:
            print_function('******************************************')
            print_function('Best model found')
            print_function('******************************************')
            print_function('Results')
            print_function('   AUC ROC   = %s' % best_model_results['AUC ROC'])
            print_function('   AUC PR    = %s' % best_model_results['AUC PR'])
            print_function('   CLL	      = %s' % best_model_results['CLL'])
            print_function('   Precision = %s at threshold = %s' % (best_model_results['Precision'][0], best_model_results['Precision'][1]))
            print_function('   Recall    = %s' % best_model_results['Recall'])
            print_function('   F1        = %s' % best_model_results['F1'])
            print_function('\n')
            print_function('Total learning time: %s seconds' % best_model_results['Learning time'])
            print_function('Total inference time: %s seconds' % best_model_results['Inference time'])
            print_function('AUC ROC: %s' % best_model_results['AUC ROC'])
        revision.delete_model_files()
        #get_saved_model_files()
        revision.delete_test_files()
        if print_function:
            print_function('Total revision time: %s' % total_revision_time)
            print_function('Best scored revision CLL: %s' % best_cll)
            print_function('\n')

        return [model, best_model_results, structured, pl_t_results]

    @staticmethod
    def get_graph(lines):
        '''Use the get_will_produced_tree function to get the WILL-Produced Tree #1
           and returns it as objects with nodes, std devs and number of examples reached.'''
        def get_match(match):
            if len(match) == 1:
                return '%.3f' % (float(match[0]))
            return '%.3f(%s)' % (float(match[0]), match[1].strip().replace('#', ''))

        lines = lines.split('\n')
        current = []
        stack = []
        target = None
        nodes = {}
        leaves = {}
        ids = {}
        last_id = 1
        graph = ''

        for line in lines:
            if not target:
                match = re.match('\s*\%\s*FOR\s*(\w+\([\w,\s]*\)):', line)
                match = re.match('\s*\%\s*FOR\s*(\w+\([\w,\s]*\)):', line)
                if match:
                    target = match.group(1)
            match = re.match('.*if\s*\(\s*([\w\(\),\s]*)\s*\).*', line)
            if match:
                nodes[','.join(current)] = match.group(1).strip()
                stack.append(current+['false'])
                current.append('true')
            match = re.match('.*[then|else] return ([\d.-]*);\s*\/\/\s*std dev\s*=\s*[\d,.\-e]*,.*\/\*\s*(.*)\s*\*\/.*', line)
            if match:
                leaves[','.join(current)] = get_match(match.groups()) #float(match.group(1))
                if len(stack):
                    current = stack.pop()
            else:
                match = re.match('.*[then|else] return ([\d.-]*);\s*\/\/\s*.*', line)
                if match:
                    leaves[','.join(current)] = get_match(match.groups()) #float(match.group(1))
                    if len(stack):
                        current = stack.pop()

        tree = revision.get_tree(nodes, leaves, [], no_variances=True)
        stack = [(1, tree)]
        last_id = 1

        while len(stack):
            t = stack.pop()
            current_id = t[0]
            root = t[1]
            if isinstance(root, str):
                graph += str(current_id) + '[shape = box,label = "' + root + '"];\n'
            elif isinstance(root, dict):
                i = list(root.keys())[0]
                value = root[i]
                children = value[1]
                true_child = children[0]
                false_child = children[1]
                last_id += 2
                stack.append((last_id, false_child))
                stack.append((last_id-1, true_child))
                graph += str(current_id) + '[label = "[' + i + ']"];\n'
                graph += str(current_id) + ' -> ' + str(last_id-1) + '[label="True"];\n'
                graph += str(current_id) + ' -> ' + str(last_id) + '[label="False"];\n'
#        for key, value in nodes.items():
#            ids[key] = last_id
#            graph += str(last_id) + '[label = "[' + value + ']"];\n'
#            last_id += 1
#        for key, value in leaves.items():
#            ids[key] = last_id
#            graph += str(last_id) + '[shape = box,label = "' + value + '"];\n'
#            last_id += 1
#        for key, value in nodes.items():
#            t = key.split(',')
#            t = [] if len(t) == 1 and t[0] == '' else t
#            current = ids[key]
#            to = ids[','.join(t + ['true'])]
#            graph += str(current) + ' -> ' + str(to) + '[label="True"];\n'
#            to = ids[','.join(t + ['false'])]
#            graph += str(current) + ' -> ' + str(to) + '[label="False"];\n'
        return 'digraph G{\n' + graph + '}'

#structured = ['interaction(A, B)', {'': 'proteinclass(B, C), proteinclass(A, C)', 'false,false,true': 'proteinclass(B, I)', 'false': 'proteinclass(A, E)', 'false,true': 'enzyme(A, F), enzyme(B, F)', 'false,false,false': 'proteinclass(B, J)', 'false,false': 'enzyme(A, H), enzyme(B, H)', 'false,true,false': 'proteinclass(B, G)', 'true': 'enzyme(A, D), enzyme(B, D)'}, {'false,false,true,false': [0.0, 0, 11], 'false,true,true': [0.0, 0, 0], 'false,false,false,false': [25.636, 1306, 1323], 'false,true,false,true': [5.148, 53, 53], 'false,false,false,true': [12.09, 307, 279], 'true,true': [0.0, 0, 4], 'true,false': [1.279, 2, 9], 'false,true,false,false': [10.986, 247, 236], 'false,false,true,true': [0.0, 0, 0]}]
#a = get_tree(structured[1], structured[2], {'': [0.11555555555555533, 0.2499979215657005], 'false,false,true': [float('nan'), -1.6148698540002277e-16], 'false': [0.24991280435603747, 0.24999999999996084], 'false,true': [float('nan'), 0.2499128043560381], 'false,true,false': [0.24999999999999997, 0.24987033250603644], 'false,false': [-1.6148698540002277e-16, 0.2499970733995874], 'false,false,false': [0.24942923039290169, 0.24998954662144235], 'true': [0.0, 0.14876033057851218]})
#print('aaa')
#b = generalize_tree(a)
