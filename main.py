#!/usr/bin/env python3 

from util import utilities    
from GFDT import GraphFragmentDecisionTree
from sklearn.model_selection import KFold
import numpy as np
import argparse
import time
import datetime
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
import pandas as pd



parser = argparse.ArgumentParser()
parser.add_argument("-data", "--dataset", type=str,  required=True)
parser.add_argument("-ncv", "--ncvfolds", type=int, default = 5,  required=False)
parser.add_argument("-rv", "--random_seed_cv", type=int, default = None,  required=False)
parser.add_argument("-ro", "--random_seed_oper", type=int, default = 1, required=False)
parser.add_argument("-d", "--depth", type=int, default = 99, required=False)
parser.add_argument("-r", "-ratio", "--saturate_ratio", type=float, default = 1.0,  required=False)
parser.add_argument("-s", "--minsup", type=int, default = 1,  required=False)
parser.add_argument("-mss", "--min_samples_split", type=int, default = 2,  required=False)


args = parser.parse_args()
dataset = args.dataset
random_seed_cv = args.random_seed_cv
ncvfolds = args.ncvfolds
max_depth = args.depth
minsup = args.minsup
min_samples_split = args.min_samples_split
saturate_ratio = args.saturate_ratio




parameters = {
    "max_depth": max_depth,
    "saturate_ratio": saturate_ratio, 
    "random_seed_oper": args.random_seed_oper,    
    "min_samples_split": min_samples_split    
}

######################################################################################
start_time = datetime.datetime.today()
logger = getLogger(__name__)
log_fmt = Formatter("%(asctime)s %(message)s")
handler = StreamHandler()
handler.setLevel("INFO")
logger.addHandler(handler)
logger.setLevel(DEBUG)

X, y = utilities.read_from_file(dataset)

X=np.array(X)
y=np.array(y)


model_info_list = []

kf = KFold(n_splits=ncvfolds, random_state=random_seed_cv, shuffle=True)
logger.info("{}CV , random_seed_cv: {}".format(ncvfolds, random_seed_cv))

for train_index, test_index in kf.split(y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model = GraphFragmentDecisionTree(**parameters)
    model.fit(X_train, y_train)        
    model_info = (model.info_tree())
    model_info_list.append(model_info)
    test_acc = model.return_accuracy(X_test, y_test)
    train_acc = model.return_accuracy(X_train, y_train)
    model_info["test_acc"] = test_acc
    model_info["train_acc"] = train_acc


info_df = pd.DataFrame(model_info_list)
info_sum = info_df.mean().to_dict()
info_sum["var_test_acc"] = info_df.var()["test_acc"]
######################################################################################


end_time = datetime.datetime.today()


logger.info("average test acc: \t{}".format(info_sum["test_acc"]))
logger.info("average train acc:\t{}".format(info_sum["train_acc"]))
logger.info("execution time of main.py: {}".format(end_time-start_time))	## by arim on 171230 