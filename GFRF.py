import numpy as np
from GFDT import GraphFragmentDecisionTree
import time

class GraphFragmentRandomForest(object):
    def __init__(self, n_estimators=5, max_depth=20, min_samples_split = 2,
                 saturate_ratio = 1.0, max_lookahead_depth = 3, bootstrap=1.0, restrict_ratio = 1.0, only_tree_option = False, random_seed_oper = None):
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.saturate_ratio=saturate_ratio
        self.max_lookahead_depth=max_lookahead_depth        
        self.bootstrap = bootstrap
        self.restrict_ratio = restrict_ratio
        self.only_tree_option = only_tree_option
        self.random_seed_oper = random_seed_oper
        
    def fit(self, X, y):
        time1 = time.time()
        self.X = X
        self.y = y 
        self.label_num = len(np.unique(y))
        
        self.tree_list = []
        self.ave_leaf_number = 0
                
        for i in range(self.n_estimators):
            tree = GraphFragmentDecisionTree(max_depth = self.max_depth, saturate_ratio =self.saturate_ratio, restrict_ratio = self.restrict_ratio, only_tree_option= self.only_tree_option, min_samples_split=self.min_samples_split, random_seed_oper = self.random_seed_oper+i)
                                                
            X_bag, y_bag =self.return_bootstrap_sample(self.X, self.y)
            tree.fit(X_bag, y_bag)            
            
            self.tree_list.append(tree)

        time2 = time.time()

        return    
        
    def return_accuracy(self, test_X, test_y):
        
        correct = 0
        wrong = 0
        
        
        for i, x in enumerate(test_X):
            
            pre_dic_list = {j:[] for j in range(self.label_num)}
            for tree in self.tree_list:                
                probability_dic = tree.return_probability(x, tree.root_node, [])
                for key in probability_dic.keys():
                    pre_dic_list[key].append(probability_dic[key])

            ave_pre_dic = {i: np.average(pre_dic_list[i]) for i in pre_dic_list.keys()}
            max_p = 0
            predict_label = None
            for key in ave_pre_dic.keys():
                if ave_pre_dic[key] > max_p:
                    max_p = ave_pre_dic[key]
                    predict_label = key
 

            if predict_label == test_y[i]:
                correct += 1
            else: wrong += 1
        return correct/(correct+wrong)
            
    def return_bootstrap_sample(self, X, y):
        if self.bootstrap == 0:            
            return X, y

        
        id_list = np.random.choice(len(self.y),int(len(self.y)*self.bootstrap))
        
        newX = [X[i] for i in id_list]
        newy = [y[i] for i in id_list]
        

        return newX, newy
        
        

