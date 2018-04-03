import numpy as np
import time
from collections import defaultdict
import random
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
import collections
from pprint import pprint
import copy 
from graphviz import Digraph



## pseudo gini score for "saturated" case
SATURATED = 9999

##=======================================================
## graph code:
## - encoding of operation code.  
## - a graph operation is encoded by 4-tuple op = (node_from, node_to, label, type_str)
# as an immutable object, where the indices are defined by 
##=======================================================

## Constants
MASTER_NODE = -1
OP_TYPE_NODE = 0 
OP_TYPE_EDGE = 1 

## Indicies of a graph code 
OP_NODE_FROM = 0	
OP_NODE_TO   = 1
OP_LABEL     = 2
OP_TYPE      = 3

class GraphFragmentDecisionTree:
    """
    Graph Fragment Decision Tree class

    """
    def __init__(self, max_depth = 100, minsup = 1, min_samples_split=2, saturate_ratio = 1.0, restrict_ratio = 1.0, only_tree_option = False, random_seed_oper = None):
        self.saturate_ratio = saturate_ratio
        self.min_samples_split = min_samples_split
        self.minsup = minsup
        self.restrict_ratio = restrict_ratio
        self.only_tree_option = only_tree_option
        self.max_depth = max_depth
        self.use_inner_node = False
        random.seed(random_seed_oper)
  
    def fit (self, X, y):
        """        
        Arguments:
        - X: a list of graph objects
        - y: a list of classification labels in numpy array
        Explanation:
        - Construct a decision tree from data X and y
        Return value: none 
        """
        self.X = X         
        self.y = y
        self.ytable = np.unique(y) #list of target labels
        self.nodelabeltable = self._collect_unique_node_labels(self.X) #list of node labels of graph dataset
        self.root_node = self.recGFDT(S = range(len(X)),  code = [], OccTable = {i: [] for i in range(len(X)) }, candidates = [(MASTER_NODE, 0, label, OP_TYPE_NODE) for label in self.nodelabeltable])
        self._post_process_decision_tree()
                
    def recGFDT(self, S,  code, OccTable, candidates):
        """ Construct a graph-fragment decision tree from a given sample """
        num_data = len(S)
        base_score = self.gini_score(S)
        
        #[1] Check termination conditions.
        if (len(code) == self.max_depth or num_data < self.min_samples_split or base_score == 0):
            return LeafNode(label_dist = self.compute_label_dist(S), support = len(S), code = code)
        
        #[2] Select an operation to split this node.
        random.shuffle(candidates)
        best_oper, best_score = self._select_best_oper(S, code, candidates, OccTable)
        
        #[3] Check termination condition of the lower bound of gain score.
        if best_score > base_score and best_score != SATURATED:
            return LeafNode(label_dist = self.compute_label_dist(S), support = len(S), code = code)
        
        if not best_oper:
            return LeafNode(label_dist = self.compute_label_dist(S), support = len(S), code = code)

        #[4] Update OccTable and operation candidates.
        S1, S0 = self.split_sample_by_oper(S, OccTable, best_oper)
        new_code = code + [best_oper]
        new_candidates = self.update_candidates_by_oper(candidates, new_code)                        
        self.update_occurrence_table_by_oper(S1, OccTable, best_oper)
        
        node0 = self.recGFDT(S0,  code, OccTable, candidates)
        node1 = self.recGFDT(S1, new_code, OccTable, new_candidates)

        return BranchNode(new_code, node0, node1, len(S), self.compute_label_dist(S))

    def _select_best_oper(self, S, code, candidates, OccTable): 
        best_score = 999
        best_oper = None
        count = 0
        if self.restrict_ratio != 1: 
            oper_score_list = []
            for oper in candidates[:]:                
                S1, S0 = self.split_sample_by_oper(S, OccTable, oper)                
                if len(S1) < self.minsup:
                    candidates.remove(oper)
                    continue
                if len(S1) >= len(S)*self.saturate_ratio:
                    score = SATURATED
                    return oper, SATURATED
                score = (len(S1)/len(S))*self.gini_score(S1) + (len(S0)/len(S))*self.gini_score(S0)
                oper_score_list.append((oper, score))
                        

            if self.restrict_ratio == 0: res_num = 1
            else: res_num = np.ceil(len(oper_score_list)*self.restrict_ratio)
          
            for i, oper_score in enumerate(oper_score_list):
                if i >= res_num: break
                if oper_score[1] < best_score:
                    best_score = oper_score[1]
                    best_oper = oper_score[0]
                else: break
                    
            return best_oper, best_score

        cand_num = len(candidates)
        for oper in candidates[:]:#candidatesは途中で要素が削除されうるためコピーを用いる
            count += 1
            S1, S0 = self.split_sample_by_oper(S, OccTable, oper)
            if len(S1) == 0:
                candidates.remove(oper)                
                continue
            if len(S1) < self.minsup: #minsup(葉のサンプル数の下限)による制限
                candidates.remove(oper)                
                continue
            if len(S1) >= len(S)*self.saturate_ratio:     
                score = SATURATED
                return oper, SATURATED
            score = (len(S1)/len(S))*self.gini_score(S1) + (len(S0)/len(S))*self.gini_score(S0)

            if score < best_score:
                if count <= cand_num*self.restrict_ratio:
                    best_score = score
                    best_oper = oper
                else: break
        return best_oper, best_score

    def update_occurrence_table_by_oper(self, S1, OccTable, oper):
        """ Returns the new occlist for a given child code from 
        the occlist for a parent code. 
        """            
        frm = oper[OP_NODE_FROM]; to = oper[OP_NODE_TO]; label = oper[OP_LABEL]; oper_type = oper[OP_TYPE]

        if frm == -1: 
            for i in S1:
                x = self.X[i]
                new_occs = []
                if label in x.search_dic:
                    for node in x.search_dic[label]:
                        new_occs.append([node])
                OccTable[i] = new_occs
            return OccTable
        
    
        for i in S1:
            x = self.X[i]
            new_occs = []            
            for occ in OccTable[i]:
                if oper_type == OP_TYPE_NODE:
                    node0 = occ[oper[OP_NODE_FROM]]            
                    label0 = oper[OP_LABEL]
                    query = (node0, label0)
                    if query in x.cache:
                        for (node1, eid1) in x.cache[query]:
                            if node1 in occ: continue
                            new_occs.append(occ + [node1])
                
                else:                            
                    v0 = occ[frm]
                    v1 = occ[to]
                    query = (v0, v1, 0)
                    if query in x.cache:
                        new_occs.append(occ)
                        
            # if len(new_occs) >= 10000: print("train", len(new_occs)) #あとで消す
            OccTable[i] = new_occs
                        
        
        return OccTable

    def split_sample_by_oper(self, S, OccTable, oper):
        S1 = []; S0 = []
        for i in S:
            if self.ismatches_by_oper(self.X[i], OccTable[i], oper):
                S1.append(i)
            else: S0.append(i)
        return S1, S0
    
    def ismatches_by_oper(self, x, occs, oper):
        
        if oper[OP_TYPE] == OP_TYPE_NODE:
            if oper[OP_NODE_FROM] == -1: # exception handling (the first node-adding operation)
                if oper[OP_LABEL] in x.search_dic:
                    return True
                else: False
                
            for occ in occs:
                node0 = occ[oper[OP_NODE_FROM]]            
                label0 = oper[OP_LABEL]
                query = (node0, label0)
                if query in x.cache:
                    for (node1, eid1) in x.cache[query]:
                        if node1 not in occ: return True
        else:
            for occ in occs:
                v0 = occ[oper[0]]
                v1 = occ[oper[1]]
                query = (v0, v1, 0)
                if query in x.cache:
                    return True
        return False

    def update_candidates_by_oper(self, candidates, code):
        """ Generate the list of graph operations to extend 
        a given parent code on an input graph """
        new_candidates = copy.deepcopy(candidates)
        
        finaloper = code[-1]
        
        if len(code) == 1: new_candidates = []

        node_num = -1
        for oper in code:
            if oper[OP_TYPE] == OP_TYPE_NODE: node_num += 1
        

        if finaloper[OP_TYPE] == OP_TYPE_NODE:
            for label in self.nodelabeltable:
                new_candidates.append((node_num, 0, label, OP_TYPE_NODE))
            if not self.only_tree_option:
                pred_node = finaloper[OP_NODE_FROM]
                for v0 in range(node_num):
                    if v0 == pred_node:continue
                    new_candidates.append((node_num, v0, 0, OP_TYPE_EDGE))
                
        
        elif oper[OP_TYPE] == OP_TYPE_EDGE:
            new_candidates.remove(finaloper)

        return new_candidates

    def compute_label_dist(self, S):      
        counter_dic = {label: 0 for label in self.ytable}
        label_dist = {label: 1/len(self.ytable) for label in self.ytable}

        if len(S) == 0: return label_dist
        for sample_id in S:                    
            counter_dic[self.y[sample_id]] += 1

        for label in self.ytable:
            p = counter_dic[label]/len(S)
            label_dist[label] = p

        return label_dist
        
    def return_accuracy(self, test_X, test_y):
        """
        return accuracy based on the prediction.

        Input: Graph Object list, target label list
        Output: accuracy
        """

        correct = 0; wrong = 0
        for i, x in enumerate(test_X):
            leafnode = self.search_leaf_by_oper(x, self.root_node, [])
            prediction = leafnode.predict_label
            if prediction == test_y[i]:
                correct += 1
            else:
                wrong +=1
        return correct/(correct + wrong)
    
    ##================================================
    ## traversing a decision tree with an input graph 
    ##================================================

    def search_leaf_by_oper(self, x, node, occs):
        
        if isinstance(node, LeafNode):            
            return node
        else:
            # recursive traversing
            test_seq = node.code
            new_occs = self._expand_occlist_by_oper(x, occs, test_seq[-1])                  
            if new_occs == []:
                return self.search_leaf_by_oper(x, node.node0, occs)
            else:
                return self.search_leaf_by_oper(x, node.node1, new_occs)

    def return_probability(self, x, node, occs):
        leafnode = self.search_leaf_by_oper(x, self.root_node, [])
        return leafnode.label_dist

    def _expand_occlist_by_oper(self, x, occs, oper):
        frm = oper[OP_NODE_FROM]; to = oper[OP_NODE_TO]; label = oper[OP_LABEL]; oper_type = oper[OP_TYPE]
        new_occs = []

        if frm == -1:
            if label in x.search_dic:
                for node in x.search_dic[label]:
                    new_occs.append([node])
            return new_occs

        for occ in occs:
            if oper_type == OP_TYPE_NODE:
                node0 = occ[oper[OP_NODE_FROM]]
                label0 = oper[OP_LABEL]
                query = (node0, label0)
                if query in x.cache:
                    for (node1, eid1) in x.cache[query]:
                        if node1 in occ: continue
                        new_occs.append(occ + [node1])
            else:
                v0 = occ[frm]
                v1 = occ[to]
                query = (v0, v1, 0)
                if query in x.cache:
                    new_occs.append(occ)

        return new_occs

    def _post_process_decision_tree(self):
        
        def dfs_fullfill(node, parent_label_dist):            
            if isinstance(node, LeafNode):
                if node.support == 0:
                    # node.label_dist = parent_label_dist                    
                    node.update_label_dist(parent_label_dist)
                
            else:                
                label_dist = node.label_dist
                dfs_fullfill(node.node0, label_dist)
                dfs_fullfill(node.node1, label_dist)

        dfs_fullfill(self.root_node, {})
        
        return

    def gini_score(self, S):
        if not S:
            return 1.0
        label_counters = {label: 0 for label in self.ytable}	#label histgrapm
        for tid in S: label_counters[self.y[tid]] += 1
        score = 1.0
        num = len(S)
        for count in label_counters.values():
            p = count / num
            score = score - p**2
        return score

    @staticmethod
    def _collect_unique_node_labels(X):
        vlbset = set()
        for x in X:
            for v in x.vertices.values():
                vlbset.add(v.vlb)
        return sorted(list(vlbset)) # must be sorted because of "set" randomness

    ##================================================
    ## for visualization
    ##================================================
    def visualize(self, filename):
        #-----------------[define a recursive function]------------------------------------------------#
        def rec_add_node(node, G, node_id):
            if isinstance(node, LeafNode) == True:
                if node.support == 0:
                    G.node(str(node_id), str(node.support), shape="circle", color="darkseagreen1", style="filled")
                else:
                    G.node(str(node_id), str(node.support), shape="circle", color="pink", style="filled")
            else:
                if node.code[-1][OP_TYPE] == OP_TYPE_EDGE:
                    G.node(str(node_id), self.pretty_graph_oper(node.code[-1]), color="skyblue1", style="filled")
                else: G.node(str(node_id), self.pretty_graph_oper(node.code[-1]),color="slategray1", style="filled")
                rec_add_node(node.node1, G, node_id*2 + 1)
                G.edge(str(node_id), str(node_id*2 + 1), str(node.node1.support))
                rec_add_node(node.node0, G, node_id*2)
                G.edge(str(node_id), str(node_id*2), str(node.node0.support))
        #-----------------------------------------------------------------------------#
        G = Digraph(format='png')
        G.attr('node', shape='box')
        rec_add_node(self.root_node, G, 1)
        G.render(filename)
    
    @staticmethod
    def pretty_graph_oper(oper):
        frm = oper[OP_NODE_FROM]; to = oper[OP_NODE_TO]; label = oper[OP_LABEL]; oper_type = oper[OP_TYPE]
        alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
        if oper_type == OP_TYPE_NODE:
            return str((frm, alphabet[int(label)]))
        else: return "E-"+str((frm, to))
    
    def info_tree(self):
        #-----------------[define a recursive function]------------------------------------------------#
        def dfs_traverse(node):
            self.num_node += 1
            if isinstance(node, LeafNode) == True:
                if node.support >= 1:
                    self.num_nonzero_leaf += 1
                    self.num_total_train_data += node.support
                    self.ave_code_length_per_data += node.support * len(node.code)
                    if len(node.code) >= self.max_code_length: self.max_code_length = len(node.code)
            else:
                dfs_traverse(node.node1)
                dfs_traverse(node.node0)
        #-----------------------------------------------------------------------------#

        info = collections.OrderedDict()
        self.num_node = 0
        self.num_nonzero_leaf = 0
        self.max_code_length = 0
        self.ave_code_length_per_data = 0
        self.num_total_train_data = 0
        dfs_traverse(self.root_node)
        
        self.ave_code_length_per_data = round(self.ave_code_length_per_data/self.num_total_train_data, 3)
        info["num_nodes"] = self.num_node
        info["num_leaves"] = int(self.num_node/2 - 1/2)
        info["num_nonzeroleaves"] = self.num_nonzero_leaf
        info["max_code_length"] = self.max_code_length
        info["ave_code_length_per_data"] = self.ave_code_length_per_data

        return info


    ##==============================================================
    ## make feature vectors using patterns that each paths represents
    ##==============================================================

    def return_feature_vectors(self, X_train, X_test, use_inner_node = False, minsup = 1):
        #-----------------[define a recursive function]------------------------------------------------#
        def pre_dfs(node):
            nonlocal id_cnt
            node.set_id(-1)

            if isinstance(node, LeafNode):
                if node.support >= minsup:
                    node.set_id(id_cnt)
                    id_cnt += 1

            else:                
                pre_dfs(node.node1)
                pre_dfs(node.node0)
                if use_inner_node:
                    node.set_id(id_cnt)
                    id_cnt += 1
                
        #-----------------[define a recursive function]------------------------------------------------#
        def dfs_traverse(node, x, occs, feature_vector):
            
            if not occs: return
    
            if isinstance(node, LeafNode): #葉ノードのとき．
                if node.node_id != -1:
                    feature_vector[node.node_id] = True
                return
            else: #内部ノードのとき
                test_seq = node.code
                new_occs = self._expand_occlist_by_oper(x, occs, test_seq[-1])
                if use_inner_node:
                    feature_vector[node.node_id] = True                        
            
                dfs_traverse(node.node0, x, occs, feature_vector)
                dfs_traverse(node.node1, x, new_occs, feature_vector)
        #----------------------[main procedure]-------------------------------------------------------#
        feature_vectors_train = []; feature_vectors_test = []
        id_cnt = 0
        pre_dfs(self.root_node)
        for x in X_train:
            feature_vector = np.zeros(id_cnt, dtype= bool)
            feature_vectors_train.append(feature_vector)
            dfs_traverse(self.root_node, x, [-1], feature_vector)
        for x in X_test:
            feature_vector = np.zeros(id_cnt, dtype= bool)
            feature_vectors_test.append(feature_vector)
            dfs_traverse(self.root_node, x, [-1], feature_vector)

        return np.array(feature_vectors_train), np.array(feature_vectors_test)
        





# classes of nodes of Graph Fragment Decision Trees.

class Node:
    def __init__(self, code, support):
        self.code = code
        self.support = support
    def set_id(self, id_cnt):
        self.node_id = id_cnt

class BranchNode(Node):
    def __init__(self, code, node0, node1, support, label_dist):
        super().__init__(code, support)    
        self.node0 = node0
        self.node1 = node1
        self.label_dist = label_dist

    
        
class LeafNode(Node):
    def __init__(self, label_dist, support, code):        
        super().__init__(code, support)
        self.label_dist = label_dist        
        max_p = 0; maxlabel = None
        for label in self.label_dist.keys():
            if label_dist[label] > max_p:
                max_p = label_dist[label]
                maxlabel = label
        if maxlabel == None:print("an error occured"); exit()
        self.predict_label = maxlabel
    
    
    def update_label_dist(self, new_label_dist):
        self.label_dist = new_label_dist        
        max_p = 0; maxlabel = None
        for label in self.label_dist.keys():
            if self.label_dist[label] > max_p:
                max_p = self.label_dist[label]
                maxlabel = label
        if maxlabel == None:print("an error occured"); exit()
        self.predict_label = maxlabel

        
#    EOF

            


        

