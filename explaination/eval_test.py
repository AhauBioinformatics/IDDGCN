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

# 逐条计算指标
metrics = []
for index, preds in tqdm(enumerate(explaine_preds),total=len(explaine_preds)):
    # 给model_preds添加反向对
    preds_flip = np.flip(preds, axis=1) #把preds正着取一次交集再反过来取一次交集，加在一起就是包含反向对的交集

    # 将三元组转换为集合
    preds_set = set(map(tuple, preds))
    preds_flip_set = set(map(tuple, preds_flip))
    gt_fold_set = set(map(tuple, gt_fold[index]))
    # 将预测结果转换为三元组的集合以便比较
    precision, recall, f1_score = calculate_metrics(preds_set,preds_flip_set, gt_fold_set)
    metrics.append((precision, recall, f1_score))

# 将指标转换为 DataFrame 方便分析
metrics_df = pd.DataFrame(metrics, columns=['Precision@5', 'Recall@5', 'F1@5'])

# 使用DataFrame的.mean()方法计算指标的总平均值
avr_prec = metrics_df['Precision@5'].mean()
avr_rec = metrics_df['Recall@5'].mean()
avr_f1 = metrics_df['F1@5'].mean()

# 显示每个条目的指标
print(metrics_df)
print(f'Total average Precision@5: {avr_prec:.4f}')
print(f'Total average Recall@5: {avr_rec:.4f}')
print(f'Total average F1@5: {avr_f1:.4f}')
# 确保文件以文本模式和'utf-8'编码打开
with open('data/explain_performance.csv', 'a', encoding='utf-8', newline='') as fa:
    writer = csv.writer(fa)

    # 如果文件是第一次打开，写入表头
    if fa.tell() == 0:
        writer.writerow(['Fold', 'explain_id', 'avr_prec','avr_rec', 'avr_f1'])

    # 写入数据
    writer.writerow([fold, 'gnn', f'{avr_prec:.4f}', f'{avr_rec:.4f}', f'{avr_f1:.4f}'])