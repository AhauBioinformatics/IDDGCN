#!/usr/bin/env python3
import sys
sys.path.append('../prediction')
import tensorflow as tf
import numpy as np
import utils1
import random as rn
import IDDGCN
import pandas as pd
import os
import time
from tqdm import tqdm
import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def get_neighbors(data_subset, node_idx):
    head_neighbors = tf.boolean_mask(data_subset, data_subset[:, 0] == node_idx)
    tail_neighbors = tf.boolean_mask(data_subset, data_subset[:, 2] == node_idx)
    neighbors = tf.concat([head_neighbors, tail_neighbors], axis=0)
    return neighbors

def get_computation_graph(head, rel, tail, data, num_relations):
    neighbors_head = get_neighbors(data, head)
    neighbors_tail = get_neighbors(data, tail)
    all_neighbors = tf.concat([neighbors_head, neighbors_tail], axis=0)
    return all_neighbors


def structure_loss(masked_adj_mats, target_ratios,epoch):

    relation_counts = [
        tf.sparse.reduce_sum(adj) if isinstance(adj, tf.SparseTensor) else tf.reduce_sum(adj)
        for adj in masked_adj_mats
    ]
    total_count = tf.reduce_sum(relation_counts)


    actual_ratios = tf.stack([count / total_count for count in relation_counts])

    loss = tf.reduce_mean(tf.square(target_ratios - actual_ratios))
    return loss

def wgnnexplainer_step(head, rel, tail, num_entities, num_relations,ADJACENCY_DATA, init_value,
                      masks, optimizer,TARGET_RATIOS,THRESHOLD,writer):
    comp_graph = get_computation_graph(head, rel, tail, ADJACENCY_DATA, num_relations)
    adj_mats = utils1.get_adj_mats(comp_graph, num_entities, num_relations)

    for epoch in range(exEpochs):

        with tf.GradientTape() as tape:
            tape.watch(masks)
            masked_adjs = [adj_mats[i] * tf.sigmoid(masks[i]) for i in range(num_relations)]
            before_pred = model([
                ALL_INDICES,
                tf.reshape(head, (1, -1)),
                tf.reshape(rel, (1, -1)),
                tf.reshape(tail, (1, -1)),
                adj_mats
            ])
            pred = model([
                ALL_INDICES,
                tf.reshape(head, (1, -1)),
                tf.reshape(rel, (1, -1)),
                tf.reshape(tail, (1, -1)),
                masked_adjs
            ])

            struct_loss = structure_loss(masked_adjs, TARGET_RATIOS,epoch)


            pred_loss = - before_pred * tf.math.log(pred + 0.00001)

            loss = (pred_loss + struct_loss)/2.


            scalar_loss = tf.squeeze(loss)
            with writer.as_default():

                tf.summary.scalar('all_loss', scalar_loss.numpy(), step=epoch)

                writer.flush()


        grads = tape.gradient(loss, masks)
        optimizer.apply_gradients(zip(grads, masks))

    current_pred = []
    current_scores = []

    for i in range(num_relations):

        mask_i = adj_mats[i] * tf.sigmoid(masks[i])

        mask_idx = mask_i.values > THRESHOLD

        non_masked_indices = tf.gather(mask_i.indices[mask_idx], [1, 2], axis=1)

        if tf.reduce_sum(non_masked_indices) != 0:
            if non_masked_indices.shape[0] is not None and non_masked_indices.shape[0] > 0:
                rel_indices = tf.cast(tf.ones((non_masked_indices.shape[0], 1)) * i, tf.int64)
            else:
                print(non_masked_indices)
            triple = tf.concat([non_masked_indices, rel_indices], axis=1)
            triple = tf.gather(triple, [0, 2, 1], axis=1)
            score_array = mask_i.values[mask_idx]
            current_pred.append(triple)
            current_scores.append(score_array)

    current_scores = tf.concat([array for array in current_scores], axis=0)

    top_k_scores = tf.argsort(current_scores, direction='DESCENDING')[0:10]

    pred_exp = tf.reshape(tf.concat([array for array in current_pred], axis=0), (-1, 3))

    pred_exp = tf.gather(pred_exp, top_k_scores, axis=0)


    for mask in masks:
        mask.assign(value=init_value)
    return pred_exp


if __name__ == '__main__':
    SEED = 24
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '0'
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    rn.seed(SEED)


    EMBEDDING_DIM = 64
    LEARNING_RATE = 0.001

    NUM_ENTITIES = 845
    NUM_RELATIONS = 4
    OUTPUT_DIM = EMBEDDING_DIM
    THRESHOLD = 0.15


    TARGET_RATIOS = tf.constant([0.4, 0.4, 0.1, 0.1], dtype=tf.float32)
    # TARGET_RATIOS = tf.constant([0.4, 0.4, 0.1, 0.1], dtype=tf.float32)
    fold = 4
    # trEpoch = 5000
    # exEpochs = 10
    # layer = 2

    for exEpochs in [10]:

        start_time = time.time()

        log_dir = "logs1/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = tf.summary.create_file_writer(log_dir)

        train2idx = pd.read_csv(fr'../datasets/prediction_datasets/mode0_fold{fold}_X_train.csv', header=0, dtype=int).values
        test2idx = pd.read_csv(fr'../datasets/explanation_datasets/test_filtered_fold{fold}.csv', header=0, dtype=int).values

        ALL_INDICES = tf.reshape(tf.range(0, NUM_ENTITIES, 1, dtype=tf.int64), (1, -1))
        all_feature_matrix = pd.read_csv(r"../datasets/prediction_datasets/feature_all_248.csv", header=None)

        model = IDDGCN.get_IDDGCN_Model(
            num_entities=NUM_ENTITIES,
            num_relations=NUM_RELATIONS,
            embedding_dim=EMBEDDING_DIM,
            output_dim=OUTPUT_DIM,
            seed=SEED,
            all_feature_matrix=all_feature_matrix,
            mode=0,
            fold=fold
        )

        model.load_weights(os.path.join(f'../datasets/prediction_datasets/weights/IDDGCN_normal/mode0_fold{fold}_epoch5000_learnRate0.001_batchsize100_embdim64_weight.h5'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
# -------------------------------------------------------
        init_value = tf.random.normal(
            (1, NUM_ENTITIES, NUM_ENTITIES),
            mean=0,
            stddev=1,
            dtype=tf.float32,
            seed=SEED)
# -----------------------------------------------------
        masks = [tf.Variable(
            initial_value=init_value,
            name='mask_' + str(i),
            trainable=True) for i in range(NUM_RELATIONS)]

        ADJACENCY_DATA = tf.concat([train2idx, test2idx], axis=0)

        del train2idx

        tf_data = tf.data.Dataset.from_tensor_slices((test2idx[:, 0], test2idx[:, 1], test2idx[:, 2])).batch(1)

        best_preds = []
        for head, rel, tail in tqdm(tf_data, total=tf_data.cardinality().numpy(), desc="Processing data"):
            current_preds = wgnnexplainer_step(head, rel, tail, NUM_ENTITIES, NUM_RELATIONS, ADJACENCY_DATA,
                                              init_value, masks, optimizer, TARGET_RATIOS, THRESHOLD, writer)

            best_preds.append(current_preds)

        best_preds = [array.numpy() for array in best_preds]

        out_preds = []
        for i in tqdm(range(len(best_preds))):
            preds_i = utils1.idx2array(best_preds[i])
            out_preds.append(preds_i)
        out_preds = np.array(out_preds, dtype=object)

        current_time = int(time.time())
        np.savez(f'../datasets/explanation_datasets/IDDGCN_explain_preds_fold{fold}.npz',preds=out_preds)

        duration = time.time()-start_time

        print(f'Time: {duration/3600}h')


        print("TARGET_RATIOS:", TARGET_RATIOS.numpy())
        print("fold:", fold)

        print("exEpochs:", exEpochs)


        print('Done.')
