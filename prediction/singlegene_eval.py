#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
import utils1
import random as rn
import tensorflow as tf
import IDDGCN,IDDGCN_newgene,IDDGCN_singlegene
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_curve
# import matplotlib.pyplot as plt

SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '0'
tf.random.set_seed(SEED)
np.random.seed(SEED)
rn.seed(SEED)
EMBEDDING_DIM = 64
OUTPUT_DIM = EMBEDDING_DIM

mode = 0
fold = 3
X_train = pd.read_csv(f"../data/data_single_KRAS/mode{mode}_fold{fold}_X_train.csv")
X_test_pos= pd.read_csv(f"../data/data_single_KRAS/mode{mode}_fold{fold}_X_test.csv",header=0,index_col=None)
#-------------------------------------------------生成负样本边
heads = X_test_pos['obj'];relations=X_test_pos['rel'];tails=X_test_pos['sbj']
num_entities=114
X_test_neg= pd.read_csv(f'../data/data_single_ABL1/mode{mode}_fold{fold}_neg_X_test.csv',header=0,index_col=0)
X_test_neg.columns = X_test_pos.columns
X_test = pd.concat([X_test_pos,X_test_neg], axis=0)
#----------------------------------------------------

NUM_ENTITIES = 114
NUM_RELATIONS = 4

ALL_INDICES = np.arange(NUM_ENTITIES).reshape(1,-1)
x_all = pd.read_csv(f"../data/data_single_ABL1/feature_ABL1_248.csv", header=None,index_col=0).values

# x_all_test = utils1.extract_feature_in_x_all(X_test,x_all)

model = IDDGCN_singlegene.get_IDDGCN_Model(
    num_entities=NUM_ENTITIES,
    num_relations=NUM_RELATIONS,
    embedding_dim=EMBEDDING_DIM,
    output_dim=OUTPUT_DIM,
    seed=SEED,
    all_feature_matrix=x_all,
    mode=mode,
    fold=fold
)

model.load_weights(os.path.join('..','data','data_single_ABL1', 'weights', f'mode{mode}_fold{fold}_epoch100_learnRate0.001_batchsize20_embdim{EMBEDDING_DIM}_weight.h5'))

ADJACENCY_DATA = tf.concat([X_train,X_test_pos], axis=0)

ADJ_MATS = utils1.get_adj_mats(ADJACENCY_DATA,NUM_ENTITIES,NUM_RELATIONS)


X_test = np.expand_dims(X_test,axis=0)
X_test_pos = np.expand_dims(X_test_pos,axis=0)

RULES = [0,1]
rel2idx = {0:0,1:1}
threshold = 0.5

for rule in RULES:
    # 选某种关系对应的索引
    rule_indices = X_test[0,:,1] == rel2idx[rule]
    X_test_rule = X_test[:, rule_indices, :]
    preds = model.predict(
        x=[
            ALL_INDICES,
            X_test_rule[:, :, 0],
            X_test_rule[:, :, 1],
            X_test_rule[:, :, 2],
            ADJ_MATS
        ]
    )

    y_pred = np.zeros_like(preds)
    y_pred[preds > threshold] = 1
    y_prob = preds[0]
    y_true = utils1.get_y_true(X_test_pos[0], X_test_rule[0])

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred[0]).ravel()
    acc = (tn + tp) / (tn + fp + fn + tp)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    specificity = tn / (tn +fp)
    f1 = 2 * precision * recall / (precision + recall)

    roc_auc = roc_auc_score(y_true, y_prob)
    prec, reca, _ = precision_recall_curve(np.array(y_true), np.array(y_prob))
    aupr = auc(reca, prec)
    # print(f' -----------relation{rule}\naccuracy:{acc:.4f}')
    # print(f'tp:{tp} | tn:{tn} | fp:{fp} | fn:{fn} | recall:{recall:.4f} | precision:{precision:.4f} | specificity:{specificity:.4f} | f1:{f1:.4f}')
    # print(f'roc_auc:{roc_auc:.4f} | aupr:{aupr:.4f}\n')

# ------------------------------------
rule_indices0 = X_test[0,:,1] == rel2idx[0]
rule_indices1 = X_test[0,:,1] == rel2idx[1]
rule_indices = np.logical_or(rule_indices0, rule_indices1)
preds = model.predict(
    x=[
        ALL_INDICES,
        X_test[:, :, 0],
        X_test[:, :, 1],
        X_test[:, :, 2],
        ADJ_MATS
    ]
)
y_pred = np.zeros_like(preds)
y_pred[preds > threshold] = 1
y_prob = preds[0]
y_true = utils1.get_y_true(X_test_pos[0], X_test[0])
tn, fp, fn, tp = confusion_matrix(y_true, y_pred[0]).ravel()
acc = (tn + tp) / (tn + fp + fn + tp)
recall = tp / (tp + fn)
precision = tp / (tp + fp)
specificity = tn / (tn + fp)
f1 = 2 * precision * recall / (precision + recall)

roc_auc = roc_auc_score(y_true, y_prob)
prec, reca, _ = precision_recall_curve(np.array(y_true), np.array(y_prob))
aupr = auc(reca, prec)
print(f' -----------relation01\naccuracy:{acc:.4f}')
print(f'tp:{tp} | tn:{tn} | fp:{fp} | fn:{fn} | recall:{recall:.4f} | precision:{precision:.4f} | specificity:{specificity:.4f}  | f1:{f1:.4f}')
print(f'roc_auc:{roc_auc:.4f} | aupr:{aupr:.4f}\n')

