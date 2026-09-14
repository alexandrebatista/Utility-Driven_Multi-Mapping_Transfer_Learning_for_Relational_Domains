'''
   Python package that makes it easier to write code that uses the BoostSRL java package, without having
   to create the data then run the jar manually.
   The code was modified to be able to learn parameteres and refine trees.

   Name:         boostsrl.py
   Author:       Alexander L. Hayes
   Modifier by:  Rodrigo Azevedo
   Updated:      July 22, 2017
   License:      GPLv3
'''

from __future__ import print_function
import os
import re
import sys

if os.name == 'posix' and sys.version_info[0] < 3:
    import subprocess32 as subprocess
else:
    import subprocess

# Mode definitions and predicate logic examples can be verified with regular expressions.
mode_re = re.compile(r'[a-zA-Z0-9]*\(((\+|\-|\#|\`)[a-zA-Z0-9]*,( )*)*(\+|\-|\#|\`)[a-zA-Z0-9]*\)\.')
exam_re = re.compile(r'[a-zA-Z0-9]*\(([a-zA-Z0-9]*,( )*)*[a-zA-Z0-9]*\)\.')

def results_to_float(string):
    '''Results can be printed with comma format.'''
    return float(string.replace(',','.'))

def examples_to_float(string):
    '''Results can be printed with comma format.'''
    return int(string.replace('.',''))

def time_to_float(string):
    '''Results can be printed with comma format.'''
    if '.' in string and ',' in string:
        return float(string.replace('.','').replace(',','.'))
    elif ',' in string:
        return float(string.replace(',','.'))
    return float(string)

def example_data(example):
    '''For demo purposes, include some sample data.
         train_pos = sample_data(train_pos)
         train_neg = sample_data(train_neg)
         train_facts = sample_data(train_facts)'''
    if example == 'train_pos':
        return ['cancer(Alice).', 'cancer(Bob).', 'cancer(Chuck).', 'cancer(Fred).']
    elif example == 'train_neg':
        return ['cancer(Dan).','cancer(Earl).']
    elif example == 'train_facts':
        return ['friends(Alice, Bob).', 'friends(Alice, Fred).', 'friends(Chuck, Bob).', 'friends(Chuck, Fred).', 'friends(Dan, Bob).', 'friends(Earl, Bob).','friends(Bob, Alice).', 'friends(Fred, Alice).', 'friends(Bob, Chuck).', 'friends(Fred, Chuck).', 'friends(Bob, Dan).', 'friends(Bob, Earl).', 'smokes(Alice).', 'smokes(Chuck).', 'smokes(Bob).']
    elif example == 'test_pos':
        return ['cancer(Zod).', 'cancer(Xena).', 'cancer(Yoda).']
    elif example == 'test_neg':
        return ['cancer(Voldemort).', 'cancer(Watson).']
    elif example == 'test_facts':
        return ['friends(Zod, Xena).', 'friends(Xena, Watson).', 'friends(Watson, Voldemort).', 'friends(Voldemort, Yoda).', 'friends(Yoda, Zod).', 'friends(Xena, Zod).', 'friends(Watson, Xena).', 'friends(Voldemort, Watson).', 'friends(Yoda, Voldemort).', 'friends(Zod, Yoda).', 'smokes(Zod).', 'smokes(Xena).', 'smokes(Yoda).']
    elif example == 'background':
        return ['friends(+Person, -Person).', 'friends(-Person, +Person).', 'smokes(+Person).', 'cancer(+Person).']
    else:
        raise(Exception('Attempted to use sample data that does not exist.'))

def call_process(call):
    '''Create a subprocess and wait for it to finish. Error out if errors occur.'''
    try:
        p = subprocess.Popen(call, shell=True)
        os.waitpid(p.pid, 0)
    except:
        raise(Exception('Encountered problems while running process: ', call))

def inspect_mode_syntax(example):
    '''Uses a regular expression to check whether all of the examples in a list are in the correct form.
       Example:
          friends(+person, -person). ::: pass
          friends(-person, +person). ::: pass
          friends(person, person).   ::: FAIL
    '''
    if not mode_re.search(example):
        raise(Exception('Error when checking background knowledge; incorrect syntax: ' + example + \
                        '\nBackground knowledge should only contain letters and numbers, of the form: ' + \
                        'predicate(+var1, -var2).'))

def inspect_example_syntax(example):
    '''Uses a regular expression to check whether all of the examples in a list are in the correct form.
       Example:
          friends(Bob, Tom).         ::: pass
          friends(+person, -person). ::: FAIL
    '''
    if not exam_re.search(example):
        raise(Exception('Error when checking example; incorrect syntax: ' + example))

def write_to_file(content, path):
    '''Takes a list (content) and a path/file (path) and writes each line of the list to the file location.'''
    with open(path, 'w') as f:
        for line in content:
            f.write(line + '\n')
    f.close()

'''
def build_bridges(target, bk):
I'm experimenting with whether bridgers can be set automatically. I'll experiment with the ability here.
    # Loop through the background information in order to find the target.
    for pred in bk:
        if (target + '(') in pred:
            # Number of commas in the predicate changes the behavior of the querypred.
            num_commas = pred.count(',')
            if num_commas > 0:
                print('querypred: ' + target + '/' + str(num_commas + 1))
                break

def save_model(model):
    Take the trees from the current model and pickle them.
    import cPickle as pickle
    pass
'''

class modes(object):

    def __init__(self, background, target, bridgers=None, precomputes=None, loadAllLibraries=False,
                 useStdLogicVariables=False, usePrologVariables=False,
                 recursion=False, lineSearch=False, resampleNegs=False,
                 treeDepth=None, maxTreeDepth=None, nodeSize=None, numOfClauses=None, numOfCycles=None, minLCTrees=None, incrLCTrees=None):
        '''
        target: a list of predicate heads that learning/inference will be performed on.
        '''
        self.target = target

        self.bridgers = bridgers
        self.precomputes = precomputes

        self.loadAllLibraries = loadAllLibraries
        self.useStdLogicVariables = useStdLogicVariables
        self.usePrologVariables = usePrologVariables
        # Note to self: check further into the difference between treeDepth and maxTreeDepth
        self.treeDepth = treeDepth
        self.maxTreeDepth = maxTreeDepth
        self.nodeSize = nodeSize
        self.numOfClauses = numOfClauses
        self.numOfCycles = numOfCycles
        self.minLCTrees = minLCTrees
        self.incrLCTrees = incrLCTrees
        self.recursion = recursion
        self.lineSearch = lineSearch
        self.resampleNegs = resampleNegs
        #self.queryPred = 'advisedby/2'

        # Many of the arguments in the modes object are optional this shows us the values of the ones that are neither false nor none

        types = {
            'background should be a list.': isinstance(background, list),
            'target should be a list.': isinstance(target, list),
            'bridgers should be a list.': isinstance(bridgers, list) or bridgers is None,
            'precomputes should be a dictionary.': isinstance(precomputes, dict) or precomputes is None,
            'loadAllLibraries should be boolean.': isinstance(loadAllLibraries, bool),
            'useStdLogicVariables should be boolean.': isinstance(useStdLogicVariables, bool),
            'usePrologVariables should be boolean.': isinstance(usePrologVariables, bool),
            'recursion should be boolean.': isinstance(recursion, bool),
            'lineSearch should be boolean.': isinstance(lineSearch, bool),
            'resampleNegs should be boolean.': isinstance(resampleNegs, bool),
            'treeDepth should be an int.': isinstance(treeDepth, int) or treeDepth is None,
            'maxTreeDepth should be an int.': isinstance(maxTreeDepth, int) or maxTreeDepth is None,
            'nodeSize should be an int.': isinstance(nodeSize, int) or nodeSize is None,
            'numOfClause should be an int.': isinstance(numOfClauses, int) or numOfClauses is None,
            'numOfCycles should be an int.': isinstance(numOfCycles, int) or numOfCycles is None,
            'minLCTrees should be an int.': isinstance(minLCTrees, int) or minLCTrees is None,
            'incrLCTrees should be an int.': isinstance(incrLCTrees, int) or incrLCTrees is None
        }

        # Force type checking for input validation Issue #5
        for type_check in types:
            if not types[type_check]:
                raise(TypeError('Error when checking type: ' +  type_check))

        relevant = [[attr, value] for attr, value in self.__dict__.items() if (value is not False) and (value is not None)]
        self.relevant = relevant

        background_knowledge = []
        for a, v in relevant:
            if (a in ['useStdLogicVariables', 'usePrologVariables'] and v == True):
                s = a + ': ' + str(v).lower() + '.'
                background_knowledge.append(s)
            elif a in ['target', 'bridgers', 'precomputes']:
                pass
            elif v == True:
                s = 'setParam: ' + a + '=' + str(v).lower() + '.'
                background_knowledge.append(s)
            else:
                s = 'setParam: ' + a + '=' + str(v) + '.'
                background_knowledge.append(s)

        for pred in background:
            inspect_mode_syntax(pred)
            background_knowledge.append('mode: ' + pred)

        if self.bridgers is not None:
            for bridger in self.bridgers:
                background_knowledge.append('bridger: ' + bridger)

        if self.precomputes is not None:
            for precompute in self.precomputes:
                background_knowledge.append(self.precomputes[precompute])
                background_knowledge.append('mode: ' + precompute)

        # Write the newly created background_knowledge to a file: background.txt
        self.background_knowledge = background_knowledge
        write_to_file(background_knowledge, 'tboostsrl/background.txt')

class train(object):

    def __init__(self, background, train_pos, train_neg, train_facts, refine=None, transfer=None, save=False, advice=False, softm=False, alpha=0.5, beta=-2, trees=1):
        '''
        background: list of strings representing background knowledge.
        '''
        self.target = background.target
        self.train_pos = train_pos
        self.train_neg = train_neg
        self.train_facts = train_facts
        self.advice = advice
        self.softm = softm
        self.alpha = alpha
        self.beta = beta
        self.trees = trees

        # Syntax checking for examples in each set.
        for example in self.train_pos:
            inspect_example_syntax(example)
        for example in self.train_neg:
            inspect_example_syntax(example)
        for example in self.train_facts:
            inspect_example_syntax(example)

        # Create train folder if it does not exist
        os.makedirs('tboostsrl/train', exist_ok=True)
        # Write train_bk
        write_to_file(['import: "../background.txt".'], 'tboostsrl/train/train_bk.txt')

        # Write refine.txt if presented
        if refine:
            write_to_file(refine, 'tboostsrl/refine.txt')

        # Write transfer.txt if presented
        if transfer:
            write_to_file(transfer, 'tboostsrl/transfer.txt')

        write_to_file(self.train_pos, 'tboostsrl/train/train_pos.txt')
        write_to_file(self.train_neg, 'tboostsrl/train/train_neg.txt')
        write_to_file(self.train_facts, 'tboostsrl/train/train_facts.txt')

        combine = '' #'-combine ' if self.trees > 1 else ''

        CALL = '(cd tboostsrl; java -jar v1-0.jar -l ' + ('-refine refine.txt ' if refine else '') + ('-transfer transfer.txt ' if transfer else '') + combine + '-train train/ -target ' + ','.join(self.target) + \
               ' -trees ' + str(self.trees) + ' > train_output.txt 2>&1)'
        call_process(CALL)

    def tree(self, treenumber, target, image=False):
        # Tree number is between 0 and the self.trees.
        if (treenumber > (self.trees - 1)):
            raise Exception('Tried to find a tree that does not exist.')
        elif image:
            '''
            Writing this with Jupyter notebooks in mind.
            '''
            from graphviz import Source
            tree_file = 'tboostsrl/train/models/bRDNs/dotFiles/WILLTreeFor_' + target + str(treenumber) + '.dot' if self.trees == 1 else 'tboostsrl/train/models/bRDNs/dotFiles/CombinedTrees' + target + '.dot'
            with open(tree_file, 'r') as f:
                tree_output = ''.join(f.read().splitlines())
            src = Source(tree_output)
            return src
        else:
            tree_file = 'tboostsrl/train/models/bRDNs/Trees/' + target + 'Tree' + str(treenumber) + '.tree'
            with open(tree_file, 'r') as f:
                tree_output = f.read()
            return tree_output

    def get_training_time(self):
        '''Return the training time as a float representing the total number of seconds seconds.'''
        with open('tboostsrl/train_output.txt', 'r') as f:
            text = f.read()
        line = re.findall(r'% Total learning time \(\d* trees\):.*', text)
        # Remove the last character "." from the line and split it on spaces.
        splitline = line[0][:-1].split()
        return splitline

    def training_time_to_float(self, splitline):
        '''Convet the string representing training time into a float representing total seconds.'''
        seconds = []
        # time with comma, is this supposed to happen?
        if 'milliseconds' in splitline:
            seconds.append((time_to_float(splitline[splitline.index('milliseconds') - 1])) / 1000)
        if 'seconds' in splitline:
            seconds.append(time_to_float(splitline[splitline.index('seconds') - 1]))
        if 'minutes' in splitline:
            seconds.append(time_to_float(splitline[splitline.index('minutes') - 1]) * 60)
        if 'hours' in splitline:
            seconds.append(time_to_float(splitline[splitline.index('hours') - 1]) * 3600)
        if 'days' in splitline:
            seconds.append(time_to_float(splitline[splitline.index('days') - 1]) * 86400)
        return sum(seconds)

    def traintime(self):
        '''Combines the get_training_time and training_time_to_float functions
           to return a float representing seconds.'''
        splitline = self.get_training_time()
        return self.training_time_to_float(splitline)

    def get_variances(self, treenumber=1):
        '''Return variances of nodes'''
        with open('tboostsrl/train/train_learn_dribble.txt', 'r') as f:
            text = f.read()
        line = re.findall(r'% Path: '+ str(treenumber-1) + ';([\w,]*)\sComparing variance: ([\d.\w\-]*) .*\sComparing variance: ([\d.\w\-]*) .*', text)
        ret = {}
        for item in line:
            ret[item[0]] = [float(item[1]), float(item[2])]
        return ret

    def get_will_produced_tree(self, treenumber=1):
        '''Return the WILL-Produced Tree'''
        combine = 'Combined' if self.trees > 1 and treenumber=='combine' else '#' + str(treenumber)
        with open('tboostsrl/train/models/WILLtheories/' + self.target[0] + '_learnedWILLregressionTrees.txt', 'r') as f:
            text = f.read()
        line = re.findall(r'%%%%%  WILL-Produced Tree '+ combine +' .* %%%%%[\s\S]*% Clauses:', text)
        splitline = (line[0].split('\n'))[2:]
        for i in range(len(splitline)):
            if splitline[i] == '% Clauses:':
                return splitline[:i-2]

    def get_structured_tree(self, treenumber=1):
        '''Use the get_will_produced_tree function to get the WILL-Produced Tree #1
           and returns it as objects with nodes, std devs and number of examples reached.'''
        def get_results(groups):
            #std dev, neg, pos
            # std dev with comma, is this supposed to happen?
            ret = [results_to_float(groups[0]), 0, 0]
            if len(groups) > 1:
                match = re.findall(r'\#pos=([\d.]*).*', groups[1])
                if match:
                    ret[2] = examples_to_float(match[0])
                match = re.findall(r'\#neg=([\d.]*)', groups[1])
                if match:
                    ret[1] = examples_to_float(match[0])
            return ret

        lines = self.get_will_produced_tree(treenumber=treenumber)
        current = []
        stack = []
        target = None
        nodes = {}
        leaves = {}

        for line in lines:
            if not target:
                match = re.match('\s*\%\s*FOR\s*(\w+\([\w,\s]*\)):', line)
                if match:
                    target = match.group(1)
            match = re.match('.*if\s*\(\s*([\w\(\),\s]*)\s*\).*', line)
            if match:
                nodes[','.join(current)] = match.group(1).strip()
                stack.append(current+['false'])
                current.append('true')
            match = re.match('.*[then|else] return .*;\s*\/\/\s*std dev\s*=\s*([\d,.\-e]*),.*\/\*\s*(.*)\s*\*\/.*', line)
            if match:
                leaves[','.join(current)] = get_results(match.groups()) #float(match.group(1))
                if len(stack):
                    current = stack.pop()
            else:
                match = re.match('.*[then|else] return .*;\s*\/\/\s*.*', line)
                if match:
                    leaves[','.join(current)] = get_results(['0'] + list(match.groups())) #float(match.group(1))
                    if len(stack):
                        current = stack.pop()
        return [target, nodes, leaves]

class test(object):

    # Possibly a partial fix to Issue #3: checking for the .aucTemp.txt.lock
    if os.path.isfile('tboostsrl/test/AUC/.aucTemp.txt.lock'):
        print('Found lock file tboostsrl/test/AUC/.aucTemp.txt.lock, removing it:')
        os.remove('tboostsrl/test/AUC/.aucTemp.txt.lock')

    def __init__(self, model, test_pos, test_neg, test_facts, trees=1):
        # Create train folder if it does not exist
        os.makedirs('tboostsrl/test', exist_ok=True)
        # Write test_bk
        write_to_file(['import: "../background.txt".'], 'tboostsrl/test/test_bk.txt')

        write_to_file(test_pos, 'tboostsrl/test/test_pos.txt')
        write_to_file(test_neg, 'tboostsrl/test/test_neg.txt')
        write_to_file(test_facts, 'tboostsrl/test/test_facts.txt')

        self.target = model.target

        CALL = '(cd tboostsrl; java -jar v1-0.jar -i -model train/models/ -test test/ -target ' + \
               ','.join(self.target) + ' -trees ' + str(trees) + ' -aucJarPath . > test_output.txt 2>&1)'
        call_process(CALL)

    def summarize_results(self):
        with open('tboostsrl/test_output.txt', 'r') as f:
            text = f.read()
        line = re.findall(r'%   AUC ROC.*|%   AUC PR.*|%   CLL.*|%   Precision.*|%   Recall.*|%   F1.*', text)
        line = [word.replace(' ','').replace('\t','').replace('%','').replace('atthreshold=',';') for word in line]

        results = {
            'AUC ROC': results_to_float(line[0][line[0].index('=')+1:]),
            'AUC PR': results_to_float(line[1][line[1].index('=')+1:]),
            'CLL': results_to_float(line[2][line[2].index('=')+1:]),
            'Precision': [results_to_float(i) for i in line[3][line[3].index('=')+1:].split(';')],
            'Recall': results_to_float(line[4][line[4].index('=')+1:]),
            'F1': results_to_float(line[5][line[5].index('=')+1:])
        }
        return results

    def float_split(self, line):
        '''Returns a list where the first item is a string and the second is a float.
           Used when returning inference results.

        Example:
           >>> test.float_split('target(pred1, pred2, pred3). 0.85691')
           ['target(pred1, pred2, pred3).', 0.85691]'''
        intermediate = line.rsplit(None, 1)
        return [intermediate[0], float(intermediate[1])]

    def inference_results(self, target):
        '''Converts BoostSRL results into a Python dictionary.'''
        results_file = 'tboostsrl/test/results_' + target + '.db'
        inference_dict = {}

        with open(results_file, 'r') as f:
            for line in f.read().splitlines():
                full = self.float_split(line)
                key_predicate = full[0]
                value_regression = full[1]
                inference_dict[key_predicate] = value_regression
        return inference_dict

    def get_testing_time(self):
        '''Return the testing time as a float representing the total number of seconds seconds.'''
        with open('tboostsrl/test_output.txt', 'r') as f:
            text = f.read()
        line = re.findall(r'% Total inference time \(\d* trees\):.*', text)
        # Remove the last character "." from the line and split it on spaces.
        splitline = line[0][:-1].split()
        return splitline

    def testing_time_to_float(self, splitline):
        '''Convet the string representing testing time into a float representing total seconds.'''
        seconds = []
        # time with comma, is this supposed to happen?
        if 'milliseconds' in splitline:
            seconds.append((time_to_float(splitline[splitline.index('milliseconds') - 1])) / 1000)
        if 'seconds' in splitline:
            seconds.append(time_to_float(splitline[splitline.index('seconds') - 1]))
        if 'minutes' in splitline:
            seconds.append(time_to_float(splitline[splitline.index('minutes') - 1]) * 60)
        if 'hours' in splitline:
            seconds.append(time_to_float(splitline[splitline.index('hours') - 1]) * 3600)
        if 'days' in splitline:
            seconds.append(time_to_float(splitline[splitline.index('days') - 1]) * 86400)
        return sum(seconds)

    def testtime(self):
        '''Combines the get_testing_time and testing_time_to_float functions
           to return a float representing seconds.'''
        splitline = self.get_testing_time()
        return self.testing_time_to_float(splitline)
