# -*- coding: utf-8 -*-
import pandas as pd
import time
from graph import Graph
import numpy as np
from graphviz import Digraph


class utilities:

    def read_from_file(dataset):

        """
        Returns:
            - X(list of Graph object)
            - y(list of class label of graph)
        """
        

        
        print("input file: ", dataset)
        f = open(dataset)        
        a = f.readlines()
        graphid = 0

        
        graph_labels = []
        Graph_list = []

        eid = 0
        maxv = 0        
        for line in a:
            if line[0] == "t":
                ll = line.split()
                if Graph_list:
                    Graph_list[-1].make_search_dic()
                G = Graph(graphid)
                Graph_list.append(G)
                if ll[3] == "-1":
                    graph_labels.append(0)
                else:graph_labels.append(int(ll[3]))
                            
                graphid += 1

            elif line[0] == "v":
                
                ll = line.split()
                vid = ll[1]
                vlb = ll[2]
                G.add_vertex(vid, vlb)
                

            elif line[0] == "e":
                
                ll = line.split()
                v0 = ll[1]
                v1 = ll[2]
                elb = ll[3]                

                G.make_adjacency_list(v0, v1, eid)
                G.make_adjacency_list(v1, v0, eid)
                G.add_edge(eid, v0, v1, elb=0)
                eid += 1


            elif line[0] == "\n": continue
        Graph_list[-1].make_search_dic()
        return Graph_list, graph_labels

    def visualize_feature_vector_info(X):

        # import seaborn
        import sys
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        

        df_X = pd.DataFrame(X)
        # for i in range(len(X[0]))
        #     for e in df_X[0]:
        #         print(e)
        feature_support_dist = []
        for i in range(len(df_X.columns)):
            # print(len(df_X.columns))
            # print(df_X[i])
            # print(df_X[i].mean)
            cnt = 0
            for e in df_X[i]:
                if e: cnt +=1
            feature_support_dist.append(cnt)
        print(feature_support_dist)
        df_feature_support_dist = pd.DataFrame(feature_support_dist)

        import matplotlib.ticker as ticker        
        plt.gca().get_yaxis().set_major_locator(ticker.MaxNLocator(integer=True))
        plt.hist(feature_support_dist, bins = max(feature_support_dist))        
        # plt.show()
        plt.ylabel('number of patterns',fontsize = 16)
        plt.xlabel('support of patterns',fontsize = 16)
        plt.title('distribution of support of discovered patterns',fontsize = 16)
        
        plt.xlim([0, 500])
        plt.ylim([0, 100])
        plt.savefig("feature_dist")

        
        exit()

    def get_giniscore_of_features(X, y):

        giniscore_list = []
        for i in range(len(X[0])):
            S1 = []; S0 = []            
            for j, x in enumerate(X):
                if x[i]: S1.append(j)
                else: S0.append(j)
            S_gini = utilities.calc_gini_impurity(y)
            S1_gini = utilities.calc_gini_impurity(y[S1])
            S0_gini = utilities.calc_gini_impurity(y[S0])
            # print(S_gini, S1_gini, S0_gini)
            giniscore = S_gini - S1_gini*(len(S1)/len(y)) - S0_gini*(len(S0)/len(y))            
            giniscore_list.append(giniscore)
        return giniscore_list
            

    def calc_gini_impurity(y):
        if len(y)==0: return 0
        classdict = {}
        gini_impurity = 1
        for e in y:
            if e not in classdict:
                classdict[e]=1
            else:
                classdict[e]+=1
        
        for c in classdict.keys():
            p = classdict[c]/len(y)
            gini_impurity -= p**2
        

        return gini_impurity
            


        
        
