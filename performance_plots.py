# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 11:38:53 2018

@author: per
"""

import pandas as pd
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier

raw_datasets = [
        'L271-90',
        'O355-90',
        'T783-90',
       'T784-90',
        'T808-90',
        'T887-90',        
        'D611-90',
        'D642-90',
        'D695-90',
        'L270-90']

 method = 'alphadist' # or 'alphadist' or 'editdist' or 'SL'
dataset_type = 'clin_meas' # or clin_meas or 'drug_pres'

def performance_over_datasets(method, dataset_type):
    
    for raw_dataset in raw_datasets:
        if method == 'alphadist':
            data = pd.read_csv('results/' + dataset_type + '/alphadist/' + raw_dataset + '-ad.csv')
        elif method == 'editdist':
            data = pd.read_csv('results/' + dataset_type + '/editdist/' + raw_dataset + '-ed.csv')
        else:
            data = pd.read_csv('results/' + dataset_type + '/SL/' + raw_dataset + '-SL.csv')
        
        np.random.seed(123)
        
        # Select column of class label
        class_labels = data['ADE']
        
        # dataset without class label
        data_no_cl = data.drop('ADE', axis = 1)
        
        # Hyperparameters set according to Bagattini et al []
        rf = RandomForestClassifier(n_estimators=100,criterion="entropy")
        
        cv = StratifiedKFold(n_splits=10)
        
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        
        for train, test in cv.split(data_no_cl, class_labels):
            pred = rf.fit(data_no_cl.iloc[train], class_labels[train]).predict(data_no_cl.iloc[test])
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(class_labels[test], pred)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        print(method, raw_dataset,'meanAUC: ', mean_auc)

performance_over_datasets(method, dataset_type)