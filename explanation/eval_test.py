import pandas as pd
import numpy as np
from tqdm import tqdm
import csv

fold = 4
top=5
exp_num=10
gt_fold = np.load(f'data/gt_filtered_fold{fold}.npz', allow_pickle=True)['result']

explaine_preds = np.load(f'data/GNNExplainer_preds_fold{fold}_epoch3000.npz', allow_pickle=True)['preds']#（3395，5，3）

# explaine_preds = np.load(f'data/explaiNE_preds_fold{fold}.npz', allow_pickle=True)['preds']

def calculate_metrics(preds_set, preds_flip_set, gt_fold_set):
    preds_list = list(preds_set)
    preds_flip_list = list(preds_flip_set)

    tp = len(set(preds_list[:top]).intersection(gt_fold_set)) + len(set(preds_flip_list[:top]).intersection(gt_fold_set))
    fp = top - tp
    fn = exp_num - (len(preds_set.intersection(gt_fold_set)) + len(preds_flip_set.intersection(gt_fold_set)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score


metrics = []
for index, preds in tqdm(enumerate(explaine_preds),total=len(explaine_preds)):

    preds_flip = np.flip(preds, axis=1)


    preds_set = set(map(tuple, preds))
    preds_flip_set = set(map(tuple, preds_flip))
    gt_fold_set = set(map(tuple, gt_fold[index]))

    precision, recall, f1_score = calculate_metrics(preds_set,preds_flip_set, gt_fold_set)
    metrics.append((precision, recall, f1_score))


metrics_df = pd.DataFrame(metrics, columns=['Precision@5', 'Recall@5', 'F1@5'])


avr_prec = metrics_df['Precision@5'].mean()
avr_rec = metrics_df['Recall@5'].mean()
avr_f1 = metrics_df['F1@5'].mean()


print(metrics_df)
print(f'Total average Precision@5: {avr_prec:.4f}')
print(f'Total average Recall@5: {avr_rec:.4f}')
print(f'Total average F1@5: {avr_f1:.4f}')

with open('data/explain_performance.csv', 'a', encoding='utf-8', newline='') as fa:
    writer = csv.writer(fa)


    if fa.tell() == 0:
        writer.writerow(['Fold', 'explain_id', 'avr_prec','avr_rec', 'avr_f1'])


    writer.writerow([fold, 'gnn', f'{avr_prec:.4f}', f'{avr_rec:.4f}', f'{avr_f1:.4f}'])