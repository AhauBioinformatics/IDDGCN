import pandas as pd
import numpy as np
from tqdm import tqdm
import csv

fold = 4
top = 5
exp_num = 10
gt_fold = np.load(f'data/gt_filtered_fold{fold}.npz', allow_pickle=True)['result']

# Load predictions for both GNNExplainer and ExplainNE
gnnexplainer_preds = np.load(f'data/GNNExplainer_preds_fold{fold}.npz', allow_pickle=True)['preds']
explaine_preds = np.load(f'data/explaiNE_preds_fold{fold}.npz', allow_pickle=True)['preds']

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

def compute_metrics_for_method(preds, method_name):
    metrics = []
    for index, pred in tqdm(enumerate(preds), total=len(preds), desc=f'Processing {method_name}'):
        preds_flip = np.flip(pred, axis=1)

        preds_set = set(map(tuple, pred))
        preds_flip_set = set(map(tuple, preds_flip))
        gt_fold_set = set(map(tuple, gt_fold[index]))

        precision, recall, f1_score = calculate_metrics(preds_set, preds_flip_set, gt_fold_set)
        metrics.append((precision, recall, f1_score))

    return metrics

# Compute metrics for both methods
gnnexplainer_metrics = compute_metrics_for_method(gnnexplainer_preds, 'GNNExplainer')
explaine_metrics = compute_metrics_for_method(explaine_preds, 'ExplainNE')

# Convert to DataFrames
gnnexplainer_metrics_df = pd.DataFrame(gnnexplainer_metrics, columns=['Precision@5', 'Recall@5', 'F1@5'])
explaine_metrics_df = pd.DataFrame(explaine_metrics, columns=['Precision@5', 'Recall@5', 'F1@5'])

# Calculate averages
avr_gnnexplainer_prec = gnnexplainer_metrics_df['Precision@5'].mean()
avr_gnnexplainer_rec = gnnexplainer_metrics_df['Recall@5'].mean()
avr_gnnexplainer_f1 = gnnexplainer_metrics_df['F1@5'].mean()

avr_explaine_prec = explaine_metrics_df['Precision@5'].mean()
avr_explaine_rec = explaine_metrics_df['Recall@5'].mean()
avr_explaine_f1 = explaine_metrics_df['F1@5'].mean()

# Print average results
print("Average Metrics for GNNExplainer")
print(f'Total average GNNExplainer Precision@5: {avr_gnnexplainer_prec:.4f}')
print(f'Total average GNNExplainer Recall@5: {avr_gnnexplainer_rec:.4f}')
print(f'Total average GNNExplainer F1@5: {avr_gnnexplainer_f1:.4f}')

print("\nAverage Metrics for ExplainNE")
print(f'Total average ExplainNE Precision@5: {avr_explaine_prec:.4f}')
print(f'Total average ExplainNE Recall@5: {avr_explaine_rec:.4f}')
print(f'Total average ExplainNE F1@5: {avr_explaine_f1:.4f}')
