#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import pandas as pd
import random
from tqdm import tqdm
from sklearn.model_selection import KFold
def get_longest_trace(data, rule):
    if rule == 'full_data':
        longest_trace = data['max_trace']
    else:
        longest_trace = data[rule + '_longest_trace']

    return longest_trace


def graded_precision_recall(
        true_exp,
        pred_exp,
        true_weight,
        unk_ent_id,
        unk_rel_id,
        unk_weight_id):
    '''
    pred_exp: numpy array without padding
    true_exp: numpy array with padding

    '''

    n = len(pred_exp)

    unk = np.array([[unk_ent_id, unk_rel_id, unk_ent_id]])

    # first compute number of triples in explanation (must exclude padded triples)
    num_explanations = 0

    # number of triples per explanation
    num_true_triples = []

    for i in range(len(true_exp)):

        current_trace = true_exp[i]

        num_triples = (current_trace != unk).all(axis=1).sum()

        if num_triples > 0:
            num_explanations += 1
            num_true_triples.append(num_triples)

    num_true_triples = np.array(num_true_triples)

    relevance_scores = np.zeros(num_explanations)

    for i in range(n):

        current_pred = pred_exp[i]

        for j in range(num_explanations):
            unpadded_traces = remove_padding_np(true_exp[j], unk_ent_id, unk_rel_id)

            unpadded_weights = true_weight[j][true_weight[j] != unk_weight_id]

            indices = (unpadded_traces == current_pred).all(axis=1)

            sum_weights = sum([float(num) for num in unpadded_weights[indices]])

            relevance_scores[j] += sum_weights

    precision_scores = relevance_scores / (n * .9)
    recall_scores = relevance_scores / (num_true_triples * .9)

    nonzero_indices = (precision_scores + recall_scores) != 0

    if np.sum(nonzero_indices) == 0:
        f1_scores = [0.0]
    else:

        nonzero_precision_scores = precision_scores[nonzero_indices]
        nonzero_recall_scores = recall_scores[nonzero_indices]

        f1_scores = 2 * (nonzero_precision_scores * \
                         nonzero_recall_scores) / (nonzero_precision_scores + nonzero_recall_scores)

    # f1_scores = 2 * (precision_scores * recall_scores) / (precision_scores + recall_scores + .000001)

    f1 = np.max(f1_scores)
    precision = np.max(precision_scores)
    recall = np.max(recall_scores)

    return precision, recall, f1


def pad_trace(trace, longest_trace, max_padding, unk):
    # unk = np.array([['UNK_ENT','UNK_REL','UNK_ENT']])

    unk = np.repeat(unk, [max_padding], axis=0)

    unk = np.expand_dims(unk, axis=0)

    while trace.shape[0] != longest_trace:
        trace = np.concatenate([trace, unk], axis=0)

    return trace


def pad_weight(trace, longest_trace, unk_weight):
    while trace.shape[0] != longest_trace:
        trace = np.concatenate([trace, unk_weight], axis=0)

    return trace


def f1(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


def jaccard_score_np(true_exp, pred_exp):
    num_true_traces = true_exp.shape[0]
    num_pred_traces = pred_exp.shape[0]

    count = 0
    for pred_row in pred_exp:
        for true_row in true_exp:
            if (pred_row == true_row).all():
                count += 1

    score = count / (num_true_traces + num_pred_traces - count)

    return score


def jaccard_score_tf(true_exp, pred_exp):
    num_true_traces = tf.shape(true_exp)[0]
    num_pred_traces = tf.shape(pred_exp)[0]

    count = 0
    for i in range(num_pred_traces):

        pred_row = pred_exp[i]

        for j in range(num_true_traces):
            true_row = true_exp[j]

            count += tf.cond(tf.reduce_all(pred_row == true_row), lambda: 1, lambda: 0)

    score = count / (num_true_traces + num_pred_traces - count)

    return score


def remove_padding_np(exp, unk_ent_id, unk_rel_id, axis=1):
    # unk = np.array(['UNK_ENT', 'UNK_REL', 'UNK_ENT'])
    unk = np.array([unk_ent_id, unk_rel_id, unk_ent_id], dtype=object)

    exp_mask = (exp != unk).all(axis=axis)

    masked_exp = exp[exp_mask]

    return masked_exp


def remove_padding_tf(exp, unk_ent_id, unk_rel_id, axis=-1):
    # unk = tf.convert_to_tensor(np.array(['UNK_ENT', 'UNK_REL', 'UNK_ENT']))
    unk = tf.cast(
        tf.convert_to_tensor([unk_ent_id, unk_rel_id, unk_ent_id]),
        dtype=exp.dtype)

    exp_mask = tf.reduce_all(tf.math.not_equal(exp, unk), axis=axis)

    masked_exp = tf.boolean_mask(exp, exp_mask)

    return masked_exp


def max_jaccard_np(current_traces, pred_exp, true_weight,
                   unk_ent_id, unk_rel_id, unk_weight_id, return_idx=False):
    ''''
    pred_exp must have shape[0] >= 1

    pred_exp: 2 dimensional (num_triples,3)

    '''

    jaccards = []
    sum_weights = []

    for i in range(len(current_traces)):
        true_exp = remove_padding_np(current_traces[i], unk_ent_id, unk_rel_id)

        weight = true_weight[i][true_weight[i] != unk_weight_id]

        sum_weight = sum([float(num) for num in weight])

        sum_weights.append(sum_weight)

        jaccard = jaccard_score_np(true_exp, pred_exp)

        jaccards.append(jaccard)

    max_indices = np.array(jaccards) == max(jaccards)

    if max_indices.sum() > 1:
        max_idx = np.argmax(max_indices * sum_weights)
        max_jaccard = jaccards[max_idx]
    else:
        max_jaccard = max(jaccards)
        max_idx = np.argmax(jaccards)

    if return_idx:
        return max_jaccard, max_idx
    return max_jaccard


def max_jaccard_tf(current_traces, pred_exp, unk_ent_id, unk_rel_id):
    '''pred_exp: 2 dimensional (num_triples,3)'''

    jaccards = []

    for i in range(len(current_traces)):
        trace = remove_padding_tf(current_traces[i], unk_ent_id, unk_rel_id)

        jaccard = jaccard_score_tf(trace, pred_exp)

        jaccards.append(jaccard)

    return max(jaccards)


def parse_ttl(file_name, max_padding):
    lines = []

    with open(file_name, 'r') as f:
        for line in f:
            lines.append(line)

    ground_truth = []
    traces = []
    weights = []

    for idx in range(len(lines)):

        if "graph us:construct" in lines[idx]:
            split_source = lines[idx + 1].split()

            source_rel = split_source[1].split(':')[1]

            source_tup = [split_source[0], source_rel, split_source[2]]

            weight = float(lines[idx + 2].split()[2][1:5])

        exp_triples = []

        if 'graph us:where' in lines[idx]:

            while lines[idx + 1] != '} \n':
                split_exp = lines[idx + 1].split()

                exp_rel = split_exp[1].split(':')[1]

                exp_triple = [split_exp[0], exp_rel, split_exp[2]]

                exp_triples.append(exp_triple)

                idx += 1

        if len(source_tup) and len(exp_triples):

            if len(exp_triples) < max_padding:

                while len(exp_triples) != max_padding:
                    pad = np.array(['UNK_ENT', 'UNK_REL', 'UNK_ENT'])
                    exp_triples.append(pad)

            ground_truth.append(np.array(source_tup))
            traces.append(np.array(exp_triples))
            weights.append(weight)

    return np.array(ground_truth), np.array(traces), np.array(weights)


def get_data(data, rule):
    # 获取french_royalty.npz中的数据
    if rule == 'full_data':

        triples = data['all_triples']
        traces = data['all_traces']
        weights = data['all_weights']

        entities = data['all_entities'].tolist()
        relations = data['all_relations'].tolist()

    else:
        triples, traces, weights = concat_triples(data, [rule])
        entities = data[rule + '_entities'].tolist()
        relations = data[rule + '_relations'].tolist()

    return triples, traces, weights, entities, relations


def concat_triples(data, rules):
    triples = []
    traces = []
    weights = []

    for rule in rules:
        triple_name = rule + '_triples'
        traces_name = rule + '_traces'
        weights_name = rule + '_weights'

        triples.append(data[triple_name])
        traces.append(data[traces_name])
        weights.append(data[weights_name])

    triples = np.concatenate(triples, axis=0)
    traces = np.concatenate(traces, axis=0)
    weights = np.concatenate(weights, axis=0)

    return triples, traces, weights


def array2idx(dataset, ent2idx, rel2idx):
    if dataset.ndim == 2:

        data = []

        for head, rel, tail in dataset:
            head_idx = ent2idx[head]
            tail_idx = ent2idx[tail]
            rel_idx = rel2idx[rel]

            data.append((head_idx, rel_idx, tail_idx))

        data = np.array(data)

    elif dataset.ndim == 3:

        data = []

        for i in range(len(dataset)):

            temp_array = []

            for head, rel, tail in dataset[i, :, :]:
                head_idx = ent2idx[head]
                tail_idx = ent2idx[tail]
                rel_idx = rel2idx[rel]

                temp_array.append((head_idx, rel_idx, tail_idx))

            data.append(temp_array)

        data = np.array(data).reshape(-1, dataset.shape[1], 3)

    elif dataset.ndim == 4:

        data = []

        for i in range(len(dataset)):

            temp_array = []

            for j in range(len(dataset[i])):

                temp_array_1 = []

                for head, rel, tail in dataset[i, j]:
                    head_idx = ent2idx[head]
                    tail_idx = ent2idx[tail]
                    rel_idx = rel2idx[rel]

                    temp_array_1.append((head_idx, rel_idx, tail_idx))

                temp_array.append(temp_array_1)

            data.append(temp_array)

        data = np.array(data)

    return data


def idx2array(dataset):
    data = []
    if dataset.ndim == 2:

        for head_idx, rel_idx, tail_idx in dataset:
            head = head_idx
            tail = tail_idx
            rel = rel_idx

            data.append((head, rel, tail))

        data = np.array(data)


    return data

# def idx2array(dataset):
#     if dataset.ndim == 2:
#
#         data = []
#
#         for head_idx, rel_idx, tail_idx in dataset:
#             head = head_idx
#             tail = tail_idx
#             rel = rel_idx
#
#             data.append((head, rel, tail))
#
#         data = np.array(data)
#
#
#     return data
def distinct(a):
    _a = np.unique(a, axis=0)
    return _a


def get_adj_mats(data, num_entities, num_relations):
    adj_mats = []

    for i in range(num_relations):

        data_i = data[data[:, 1] == i]

        if not data_i.shape[0]:
            indices = tf.zeros((1, 2), dtype=tf.int64)
            values = tf.zeros((indices.shape[0]))

        else:

            # indices = tf.concat([
            #         tf.gather(data_i,[0,2],axis=1),
            #         tf.gather(data_i,[2,0],axis=1)],axis=0)
            indices = tf.gather(data_i, [0, 2], axis=1)
            # 即抽出实体的id
            indices = tf.py_function(distinct, [indices], indices.dtype)
            indices = tf.dtypes.cast(indices, tf.int64)
            values = tf.ones((indices.shape[0]))

        sparse_mat = tf.sparse.SparseTensor(

            indices=indices,
            values=values,
            dense_shape=(num_entities, num_entities)
        )
        sparse_mat = tf.sparse.reorder(sparse_mat)

        sparse_mat = tf.sparse.reshape(sparse_mat, shape=(1, num_entities, num_entities))
        adj_mats.append(sparse_mat)
    return adj_mats


def get_negative_triples(head, rel, tail, num_entities, seed):
    cond = tf.random.uniform(tf.shape(head), 0, 2, dtype=tf.int64, seed=seed)  # 相当于是生成掩码
    rnd = tf.random.uniform(tf.shape(head), 0, num_entities - 1, dtype=tf.int64, seed=seed)

    neg_head = tf.where(cond == 1, head, rnd)  # 条件为真返回head，为假返回rnd
    neg_tail = tf.where(cond == 1, rnd, tail)
    return neg_head, neg_tail



def generate_negative_sample(head, tail, num_entities,seed):

    replace_head = tf.random.uniform([], 0, 2, dtype=tf.int32,seed=seed)
    random_entity = tf.random.uniform([], 0, num_entities, dtype=tf.int64,seed=seed)

    neg_head = tf.where(replace_head==1, random_entity, head)
    neg_tail = tf.where(replace_head==1, tail, random_entity)

    return neg_head, neg_tail

@tf.function
def get_negative_triples1(head, rel, tail, num_entities,seed):
    combined = tf.stack([head, rel, tail], axis=1)
    # Initialize negative samples tensor array
    negative_samples = tf.TensorArray(dtype=tf.int64, size=tf.shape(head)[0], dynamic_size=True)

    # Define loop condition
    def cond(i, _):
        return i < tf.shape(head)[0]

    # Loop body to generate negative samples not overlapping with positive samples
    def body(i, neg_samples):
        def generate_unique_negative_sample(head, tail):
            neg_head, neg_tail = generate_negative_sample(head, tail, num_entities,seed)
            neg_sample = tf.stack([neg_head, rel[i], neg_tail])
            is_unique = tf.logical_not(tf.reduce_any(tf.reduce_all(tf.equal(combined, neg_sample), axis=1)))
            return neg_head, neg_tail, is_unique


        neg_head, neg_tail, is_unique = generate_unique_negative_sample(head[i], tail[i])

        # Loop until a unique negative sample is found
        while_condition = lambda nh, nt, unique: tf.logical_not(unique)
        body_function = lambda nh, nt, unique: generate_unique_negative_sample(nh, nt)
        neg_head, neg_tail, _ = tf.while_loop(
            while_condition,
            body_function,
            loop_vars=[neg_head, neg_tail, is_unique],
            shape_invariants=[neg_head.get_shape(), neg_tail.get_shape(), is_unique.get_shape()]
        )

        # Write the unique negative sample to TensorArray
        neg_samples = neg_samples.write(i, tf.stack([neg_head, rel[i], neg_tail], axis=0))
        return i + 1, neg_samples

    # Execute the loop
    _, negative_samples = tf.while_loop(cond, body, loop_vars=[0, negative_samples])

    # Convert TensorArray to Tensor
    negative_samples = negative_samples.stack()
    neg_head, neg_rel, neg_tail = tf.unstack(negative_samples, axis=1)

    return neg_head, neg_tail


def train_test_split_no_entity(all_tripls,test_entities_size,seed):

    random.seed(seed)


    similar_mu = set(all_tripls[(all_tripls[1] == 3) & (all_tripls[0] < 95)][0]).union(set(all_tripls[(all_tripls[1] == 3) & (all_tripls[2] < 95)][2]))
    similar_drugs = set(all_tripls[(all_tripls[1] == 2) & (all_tripls[0] >= 95)][0]).union(set(all_tripls[(all_tripls[1] == 2) & (all_tripls[2] >= 95)][2]))
    similar = similar_drugs.union(similar_mu)

    triples_to_select = all_tripls[(all_tripls[1].isin([0, 1])) & (all_tripls[0].isin(list(similar))) & (all_tripls[2].isin(list(similar)))]
    triples_remaining = all_tripls[~all_tripls.index.isin(triples_to_select.index)]


    unique_nodes = pd.concat([triples_to_select[2],triples_to_select[0]]).unique()
    unique_mu = unique_nodes[unique_nodes<95]
    unique_drugs = unique_nodes[unique_nodes>=95]

    mu_num_to_select = int(unique_mu.shape[0] * test_entities_size)
    selected_mu = random.sample(list(unique_mus), mu_num_to_select)

    drug_num_to_select = int(unique_drugs.shape[0] * test_entities_size)
    selected_drugs = random.sample(list(unique_drugs), drug_num_to_select)

    mu_test_rows = triples_to_select[triples_to_select[0].isin(selected_mu) | triples_to_select[2].isin(selected_mu)]
    drug_test_rows = triples_to_select[triples_to_select[0].isin(selected_drugs) | triples_to_select[2].isin(selected_drugs)]
    cd_test_rows = pd.concat([mu_test_rows,drug_test_rows],axis=0).drop_duplicates()


    mu_train_dataset = triples_to_select[~triples_to_select.isin(mu_test_rows)].dropna().astype('int64')
    drug_train_dataset = triples_to_select[~triples_to_select.isin(drug_test_rows)].dropna().astype('int64')
    cd_train_dataset = triples_to_select[~triples_to_select.isin(cd_test_rows)].dropna().astype('int64')

    mu_test_dataset = pd.concat([mu_test_rows, triples_remaining])
    drug_test_dataset = pd.concat([drug_test_rows, triples_remaining])
    cd_test_dataset = pd.concat([cd_test_rows,triples_remaining])

    mu_train_dataset = pd.concat([mu_train_dataset, triples_remaining])
    drug_train_dataset = pd.concat([drug_train_dataset, triples_remaining])
    cd_train_dataset = pd.concat([cd_train_dataset, triples_remaining])

    return mu_train_dataset,mu_test_dataset,drug_train_dataset,drug_test_dataset,cd_train_dataset,cd_test_dataset

def train_test_split_no_unseen(all_tripls,test_size,seed):

    np.random.seed(seed)
    X_test = []
    if test_size==0:
        return all_tripls,X_test

    all_tripls_np = all_tripls.values

    num_test = int(len(all_tripls_np) * test_size)
    num_train = len(all_tripls_np) - num_test

    X_train = []
    X_test = []

    entities_train = set()
    relations_train = set()


    np.random.shuffle(all_tripls_np)


    for triple in all_tripls_np:
        head, relation, tail = triple

        if len(X_test) < num_test and head in entities_train and tail in entities_train and relation in relations_train:
            X_test.append(triple)
        elif len(X_train) < num_train:
            X_train.append(triple)
            entities_train.add(head)
            entities_train.add(tail)
            relations_train.add(relation)
    while len(X_test) < num_test:
        if len(X_test) > 0:
            X_test.append(X_test[np.random.randint(len(X_test))])
        else:
            break
    X_train = pd.DataFrame(np.array(X_train),columns=[0,1,2])
    X_test = pd.DataFrame(np.array(X_test),columns=[0,1,2])

    return X_train, X_test

def extract_feature_in_x_all(X,all_feature_matrix):
    nodes_to_extract = set()
    for triplet in X.values:
        nodes_to_extract.add(triplet[0])
        nodes_to_extract.add(triplet[2])
    x_all_new = []
    for node in nodes_to_extract:
        x_all_new.append(all_feature_matrix[node])

    x_all_new = pd.DataFrame(x_all_new)
    return  x_all_new


def generate_reverse_triplets(triplets):

    reverse_triplets = [(tail, relation, head) for head, relation, tail in triplets if head != tail]


    triplets_array = np.array(triplets)
    reverse_triplets_array = np.array(reverse_triplets)
    return reverse_triplets_array


def save_adjmat(ADJ_MATS):
    for i, adj_mat in enumerate(ADJ_MATS):
        dense_mat = tf.sparse.to_dense(adj_mat)

        dense_mat_value = dense_mat.numpy()

        dense_mat_value_2d = np.reshape(dense_mat_value, (dense_mat_value.shape[1], -1))

        np.savetxt(f'../data1/sparse_matrix{i}.csv', dense_mat_value_2d, delimiter=',')
def extract_feature_in_x_all(X,x_all):
    nodes_to_extract = set()
    for triplet in X.values:
        nodes_to_extract.add(triplet[0])
        nodes_to_extract.add(triplet[2])
    x_all_new = []
    for node in nodes_to_extract:
        x_all_new.append(x_all[node])

    x_all_new = pd.DataFrame(x_all_new)
    return x_all_new
def generate_negative_samples_np(heads, relations, tails, num_entities, seed):

    np.random.seed(seed)
    condition_mask = np.random.randint(0, 2, size=heads.shape)
    random_entities = np.random.randint(0, num_entities, size=heads.shape)

    neg_heads = np.where(condition_mask == 0, heads, random_entities)
    neg_tails = np.where(condition_mask == 1, tails, random_entities)

    return neg_heads, relations, neg_tails

def get_y_true(X_test_pos, X_test_rule):
    X_test_pos = pd.DataFrame(X_test_pos)
    X_test_rule = pd.DataFrame(X_test_rule)
    X_test_pos = X_test_pos.drop_duplicates()
    X_merged = pd.merge(X_test_rule, X_test_pos, indicator=True, how='left')
    y_true = (X_merged['_merge'] == 'both').astype(int)
    return y_true.values


def static_data(triples):
    count_relation_0 = triples[triples['1'] == 0].shape[0]
    count_relation_1 = triples[triples['1'] == 1].shape[0]
    count_relation_2 = triples[triples['1'] == 2].shape[0]
    count_relation_3 = triples[triples['1'] == 3].shape[0]

    print("0:", count_relation_0)
    print("1:", count_relation_1)
    print("2:", count_relation_2)
    print("3:", count_relation_3)
    return count_relation_0, count_relation_1, count_relation_2, count_relation_3
def static_data1(triples):
    count_relation_0 = triples[triples[1] == 0].shape[0]
    count_relation_1 = triples[triples[1] == 1].shape[0]
    count_relation_2 = triples[triples[1] == 2].shape[0]
    count_relation_3 = triples[triples[1] == 3].shape[0]

    print("0:", count_relation_0)
    print("1:", count_relation_1)
    print("2:", count_relation_2)
    print("3:", count_relation_3)
    return count_relation_0, count_relation_1, count_relation_2, count_relation_3
def generate_negative_samples_corresponding(positive_samples, seed, negative_per_positive):
    random.seed(seed)
    positive_samples.columns = ['mu', 'relation', 'drug']

    mu = list(set([entity for entity in positive_samples['mu'].tolist() + positive_samples['drug'].tolist() if
                      entity <= 660]))
    drugs = list(set([entity for entity in positive_samples['mu'].tolist() + positive_samples['drug'].tolist() if
                      entity >= 661]))
    mu_drugs = mu + drugs

    negative_samples = []
    positive_samples = positive_samples.to_numpy()

    for object, relation, subject in tqdm(positive_samples):
        negatives_for_sample = 0
        count = 0
        while negatives_for_sample < negative_per_positive:
            replace_entity = np.random.choice([0, 1])
            if count<50 and replace_entity == 0:
                new_sample = list((random.choice(mu_drugs), relation, subject))
            elif count<50:
                new_sample = list((object, relation, random.choice(mu_drugs)))
            else:
                new_sample = list((random.choice(mu_drugs),relation,random.choice(mu_drugs)))
            count = count+1
            if not np.any(np.all(new_sample == positive_samples, axis=1)) and new_sample not in negative_samples:
                negative_samples.append(new_sample)
                negatives_for_sample += 1

    negative_samples_df = pd.DataFrame(negative_samples, columns=['mu', 'relation', 'drug'])
    return negative_samples_df

def find_corr_optimized(target_triples, corr_edge):

    merged = pd.merge(target_triples, corr_edge, how='left', on=['obj', 'rel', 'sbj'])


    columns_neg = ['neg_obj', 'neg_rel', 'neg_sbj']
    columns_syn = ['syn_obj', 'syn_rel', 'syn_sbj']
    columns_syn_neg = ['syn_neg_obj', 'syn_neg_rel', 'syn_neg_sbj']


    neg_trans_list = merged[columns_neg]
    syn_trans_list = merged[columns_syn]
    syn_neg_trans_list = merged[columns_syn_neg]

    return neg_trans_list, syn_trans_list, syn_neg_trans_list
def find_corr_optimized_new(target_triples, corr_edge):
    merged = pd.merge(target_triples, corr_edge, how='left', on=['obj', 'rel', 'sbj'])
    columns_neg = ['neg_obj', 'neg_rel', 'neg_sbj']
    neg_trans_list = merged[columns_neg]

    return neg_trans_list
def split_pos_triple_into_folds(dc, cc, dd, num_folds, seed, mode):

    dc = dc.sample(frac=1, random_state=seed).reset_index(drop=True)
    cc = cc.sample(frac=1, random_state=seed).reset_index(drop=True)
    dd = dd.sample(frac=1, random_state=seed).reset_index(drop=True)
    dc_cc_dd = pd.concat([dc, dd, cc], axis=0).astype(int)
    cc_dd = pd.concat([cc,dd],axis=0)
    if mode==0:
        kf = KFold(n_splits=num_folds)
        train_test_splits = []
        for train_index, test_index in kf.split(dc):
            train_data, test_data = dc.iloc[train_index], dc.iloc[test_index]
            train_data = pd.concat([train_data,cc_dd],axis=0)
            train_test_splits.append((train_data, test_data))

    elif mode==1:
        train_test_splits = []
        num_entity = 660
        len_fold = num_entity//num_folds
        np.random.seed(seed)
        for i in range(num_folds):
            nodes = range(int(len_fold*i), int(len_fold*(i+1)))
            test_df = dc_cc_dd[(dc_cc_dd['obj'].isin(nodes)) | (dc_cc_dd['sbj'].isin(nodes))].astype(int)

            merged_df = pd.merge(dc_cc_dd, test_df, how='left', indicator=True)
            train_df = merged_df[merged_df['_merge'] == 'left_only'].drop('_merge', axis=1).astype(int)

            train_test_splits.append((train_df, test_df))

    elif mode==2:
        train_test_splits = []
        num_entity = 157
        len_fold = num_entity/num_folds
        np.random.seed(seed)

        for i in range(num_folds):
            nodes = range(int(len_fold*i+660), int(len_fold*(i+1)+660))
            test_df = dc_cc_dd[(dc_cc_dd['obj'].isin(nodes)) | (dc_cc_dd['sbj'].isin(nodes))].astype(int)
            merged_df = pd.merge(dc_cc_dd, test_df, how='left', indicator=True)
            train_df = merged_df[merged_df['_merge'] == 'left_only'].drop('_merge', axis=1).astype(int)
            # train_df = dc_cc_dd[~dc_cc_dd.isin(test_df)].dropna().astype(int)
            train_test_splits.append((train_df, test_df))
    else:
        train_test_splits = []
        num_entity = 660
        len_fold = num_entity/num_folds
        np.random.seed(seed)
        for i in range(num_folds):
            nodes = range(int(len_fold*i), int(len_fold*(i+1)))

            pre_test_df = dc_cc_dd[(dc_cc_dd['obj'].isin(nodes)) | (dc_cc_dd['sbj'].isin(nodes))]

            merged_df = pd.merge(dc_cc_dd, pre_test_df, how='left', indicator=True)
            pre_train_df = merged_df[merged_df['_merge'] == 'left_only'].drop('_merge', axis=1).astype(int)

            all_nodes = np.unique(pre_test_df[['obj', 'sbj']].values)
            new_node = list(np.setdiff1d(all_nodes, nodes))

            new_node_test = new_node[:len(new_node)//2]
            new_node_train = new_node[len(new_node)//2:]

            test_df = pre_test_df[~pre_test_df['obj'].isin(new_node_test) & ~pre_test_df['sbj'].isin(new_node_test)].astype(int)
            train_df = pre_train_df[~pre_train_df['obj'].isin(new_node_train) & ~pre_train_df['sbj'].isin(new_node_train)].astype(int)

            train_test_splits.append((train_df, test_df))

    return train_test_splits
def split_neg_triple_into_folds(dc, num_folds, seed, mode):

    if mode==0:
        kf = KFold(n_splits=num_folds)
        train_test_splits = []
        for train_index, test_index in kf.split(dc):
            train_data, test_data = dc.iloc[train_index], dc.iloc[test_index]
            train_test_splits.append((train_data, test_data))

    elif mode==1:
        train_test_splits = []
        num_entity = 477
        len_fold = num_entity//num_folds
        np.random.seed(seed)
        for i in range(num_folds):
            nodes = range(int(len_fold*i), int(len_fold*(i+1)))
            test_df = dc[(dc['obj'].isin(nodes)) | (dc['sbj'].isin(nodes))].astype(int)

            merged_df = pd.merge(dc, test_df, how='left', indicator=True)
            train_df = merged_df[merged_df['_merge'] == 'left_only'].drop('_merge', axis=1).astype(int)

            # train_df = dc_cc_dd[~dc_cc_dd.isin(test_df)].dropna().astype(int)
            train_test_splits.append((train_df, test_df))

    elif mode==2:
        train_test_splits = []
        num_entity = 157
        len_fold = num_entity/num_folds
        np.random.seed(seed)

        for i in range(num_folds):
            nodes = range(int(len_fold*i+477), int(len_fold*(i+1)+477))
            # nodes = list(range(477,492))
            test_df = dc[(dc['obj'].isin(nodes)) | (dc['sbj'].isin(nodes))].astype(int)
            merged_df = pd.merge(dc, test_df, how='left', indicator=True)
            train_df = merged_df[merged_df['_merge'] == 'left_only'].drop('_merge', axis=1).astype(int)
            # train_df = dc_cc_dd[~dc_cc_dd.isin(test_df)].dropna().astype(int)
            train_test_splits.append((train_df, test_df))
    else:
        train_test_splits = []
        num_entity = 477
        len_fold = num_entity/num_folds
        np.random.seed(seed)
        for i in range(num_folds):
            nodes = range(int(len_fold*i), int(len_fold*(i+1)))

            pre_test_df = dc[(dc['obj'].isin(nodes)) | (dc['sbj'].isin(nodes))]

            merged_df = pd.merge(dc, pre_test_df, how='left', indicator=True)
            pre_train_df = merged_df[merged_df['_merge'] == 'left_only'].drop('_merge', axis=1).astype(int)
            # pre_train_df = dc_cc_dd[~dc_cc_dd.isin(pre_test_df)].dropna().astype(int)

            all_nodes = np.unique(pre_test_df[['obj', 'sbj']].values)
            new_node = list(np.setdiff1d(all_nodes, nodes))
            new_node_test = new_node[:len(new_node)//2]
            new_node_train = new_node[len(new_node)//2:]
            test_df = pre_test_df[~pre_test_df['obj'].isin(new_node_test) & ~pre_test_df['sbj'].isin(new_node_test)].astype(int)
            train_df = pre_train_df[~pre_train_df['obj'].isin(new_node_train) & ~pre_train_df['sbj'].isin(new_node_train)].astype(int)
            train_test_splits.append((train_df, test_df))
    return train_test_splits
