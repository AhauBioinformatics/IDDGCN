#!/usr/bin/env python3
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from tensorflow.keras.layers import Embedding, Lambda
import utils1
import pandas as pd
import os
import numpy as np
import random as rn
from datetime import datetime
from keras.callbacks import Callback,EarlyStopping

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class IDDGCN_Layer(tf.keras.layers.Layer):
    def __init__(self, num_entities, num_relations, output_dim, seed, **kwargs):
        super(IDDGCN_Layer, self).__init__(**kwargs)
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.output_dim = output_dim
        self.seed = seed


        self.relation_kernel = self.add_weight(
            shape=(self.num_relations, self.output_dim, self.output_dim),
            name="relation_kernels",
            trainable=True,
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=self.seed)
        )
        self.self_kernel = self.add_weight(
            shape=(self.output_dim, self.output_dim),
            name="self_kernel",
            trainable=True,
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=self.seed)
        )


        self.relation_weights = self.add_weight(
            shape=(self.num_relations,),
            initializer='uniform',
            trainable=True,
            name='relation_weights'
        )


        self.W_alpha = self.add_weight(
            shape=(self.output_dim, self.num_relations),
            initializer='glorot_uniform',
            trainable=True,
            name='W_alpha'
        )
        self.b_alpha = self.add_weight(
            shape=(self.num_relations,),
            initializer='zeros',
            trainable=True,
            name='b_alpha'
        )

    def call(self, inputs):
        embeddings, head_idx, head_e, tail_idx, tail_e, *adj_mats = inputs
        head_output = tf.matmul(head_e, self.self_kernel)
        tail_output = tf.matmul(tail_e, self.self_kernel)


        alpha = tf.nn.softmax(tf.matmul(head_e, self.W_alpha) + self.b_alpha)

        for i in range(self.num_relations):
            adj_i = tf.sparse.reshape(adj_mats[0][i], shape=(self.num_entities, self.num_entities))
            sum_embeddings = tf.sparse.sparse_dense_matmul(adj_i, embeddings)
            head_update = tf.nn.embedding_lookup(sum_embeddings, head_idx)
            tail_update = tf.nn.embedding_lookup(sum_embeddings, tail_idx)

            relation_weight = tf.sigmoid(alpha[:, i])
            head_output += tf.expand_dims(relation_weight, 1) * tf.matmul(head_update, self.relation_kernel[i])
            tail_output += tf.expand_dims(relation_weight, 1) * tf.matmul(tail_update, self.relation_kernel[i])

        return tf.sigmoid(head_output), tf.sigmoid(tail_output)


class DistMult(tf.keras.layers.Layer):

    def __init__(self, num_relations, seed, **kwargs):
        super(DistMult, self).__init__(**kwargs)
        self.num_relations = num_relations
        self.seed = seed

    def build(self, input_shape):
        embedding_dim = input_shape[0][-1]

        self.kernel = self.add_weight(
            shape=(self.num_relations, embedding_dim),
            trainable=True,
            initializer=tf.keras.initializers.RandomNormal(
                mean=0.0,
                stddev=1,
                seed=self.seed
            ),
            name='rel_embedding'
        )

    def call(self, inputs):
        head_e, rel_idx, tail_e = inputs

        rel_e = tf.nn.embedding_lookup(self.kernel, rel_idx)

        score = tf.sigmoid(tf.reduce_sum(head_e * rel_e * tail_e, axis=-1))
        return tf.expand_dims(score, axis=0)


class IDDGCN_Model(tf.keras.Model):

    def __init__(self, num_entities,seed,mode,fold,neg_weight=1.0, *args, **kwargs):
        super(IDDGCN_Model, self).__init__(*args, **kwargs)
        self.num_entities = num_entities
        self.seed = seed
        self.mode = mode
        self.fold = fold
        self.neg_weight = neg_weight

    def train_step(self, data):
        all_indices, pos_head, rel, pos_tail, *adj_mats = data[0]
        y_pos_true = data[1]
        X_train_neg = np.load(f'../data/data_single_ABL1/mode{self.mode}_fold{self.fold}_X_train_neg.npy')

        num_samples = X_train_neg.shape[1]


        random_indices = np.random.permutation(num_samples)

        X_train_neg = X_train_neg[:, random_indices, :]
        neg_head = X_train_neg[:,:,0]
        neg_tail = X_train_neg[:,:,2]
        neg_rel = X_train_neg[:,:,1]

        neg_head = tf.convert_to_tensor(neg_head, dtype=tf.int64)
        neg_tail = tf.convert_to_tensor(neg_tail, dtype=tf.int64)
        neg_rel = tf.convert_to_tensor(neg_rel, dtype=tf.int64)


        with tf.GradientTape() as tape:
            y_pos_pred = self([
                all_indices,
                pos_head,
                rel,
                pos_tail,
                adj_mats
            ],training=True)
            y_neg_pred = self([
                all_indices,
                neg_head,
                neg_rel,
                neg_tail,
                adj_mats
            ],training=True)
            y_pred = tf.concat([y_pos_pred, y_neg_pred], axis=1)
            y_true = tf.concat([y_pos_true, tf.zeros_like(y_neg_pred)], axis=1)
            tf.print(y_pred)
            # tf.print(pos_head,summarize=100)
            tf.print(y_true)
            loss = self.compiled_loss(y_true, y_pred)
            tf.print(loss)
            loss *= (1 / self.num_entities)



            grads = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.compiled_metrics.update_state(y_pos_true, y_pos_pred)
            return {m.name: m.result() for m in self.metrics}


class SaveWeightsCallback(Callback):
    def __init__(self, save_epochs, save_path_template, mode, fold, learning_rate, batch_size, EMBEDDING_DIM):
        super(SaveWeightsCallback, self).__init__()
        self.save_epochs = save_epochs
        self.save_path_template = save_path_template
        self.mode = mode
        self.fold = fold
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.EMBEDDING_DIM = EMBEDDING_DIM

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 in self.save_epochs:
            filename = self.save_path_template.format(mode=self.mode, fold=self.fold, epoch=epoch + 1,
                                                      learning_rate=self.learning_rate, batch_size=self.batch_size,
                                                      EMBEDDING_DIM=self.EMBEDDING_DIM)
            self.model.save_weights(filename)
            print(f"\nSaved weights for epoch {epoch + 1} to {filename}")

def get_IDDGCN_Model(num_entities,num_relations, embedding_dim, output_dim, seed, all_feature_matrix,mode,fold):
    head_input = tf.keras.Input(shape=(None,), name='head_input', dtype=tf.int64)
    rel_input = tf.keras.Input(shape=(None,), name='rel_input', dtype=tf.int64)
    tail_input = tf.keras.Input(shape=(None,), name='tail_input', dtype=tf.int64)
    all_entities = tf.keras.Input(shape=(None,), name='all_entities', dtype=tf.int64)

    adj_inputs = [tf.keras.Input(
        shape=(num_entities, num_entities),
        dtype=tf.float32,
        name='adj_inputs_' + str(i),
        sparse=True,
    ) for i in range(num_relations)]


    entity_embeddings = Embedding(
        input_dim=num_entities,
        output_dim=embedding_dim,
        name='entity_embeddings',
        trainable=True,
        embeddings_initializer=tf.keras.initializers.RandomUniform(
             minval=0,
             maxval=1,
             seed=seed)
        )
    head_e = entity_embeddings(head_input)
    tail_e = entity_embeddings(tail_input)
    all_e = entity_embeddings(all_entities)
    head_e = Lambda(lambda x: x[0, :, :])(head_e)
    tail_e = Lambda(lambda x: x[0, :, :])(tail_e)
    all_e = Lambda(lambda x: x[0, :, :])(all_e)
    head_index = Lambda(lambda x: x[0, :])(head_input)
    rel_index = Lambda(lambda x: x[0, :])(rel_input)
    tail_index = Lambda(lambda x: x[0, :])(tail_input)

    new_head_1, new_tail_1 = IDDGCN_Layer(
        num_relations=num_relations,
        num_entities=num_entities,
        output_dim=output_dim,
        seed=seed)([
        all_e,
        head_index,
        head_e,
        tail_index,
        tail_e,
        adj_inputs])

    new_head_2, new_tail_2 = IDDGCN_Layer(
        num_relations=num_relations,
        num_entities=num_entities,
        output_dim=output_dim,
        seed=seed)([
        all_e,
        head_index,
        new_head_1,
        tail_index,
        new_tail_1,
        adj_inputs])

    new_head_3, new_tail_3 = IDDGCN_Layer(
        num_relations=num_relations,
        num_entities=num_entities,
        output_dim=output_dim,
        seed=seed)([
        all_e,
        head_index,
        new_head_2,
        tail_index,
        new_tail_2,
        adj_inputs])
    output = DistMult(num_relations=num_relations, seed=seed, name='DistMult')([new_head_3, rel_index, new_tail_3])

    model = IDDGCN_Model(
        inputs=[all_entities, head_input, rel_input, tail_input] + adj_inputs,
        outputs=[output],
        num_entities=num_entities,
        seed=seed,
        mode=mode,
        fold=fold
    )
    return model
def print_dense_matrix(sparse_tensor):
    dense_matrix = tf.sparse.to_dense(sparse_tensor)
    print(dense_matrix.numpy())

if __name__ == '__main__':

    SEED = 89
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '0'
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    rn.seed(SEED)
    # 网格调参法
    batch_size = [20]
    learning_rate = [0.001]
    embedding_dim = [64]
    save_epochs = [100,200,500,1000]

    for bs in batch_size:
        for lr in learning_rate:
            for ed in embedding_dim:

                BATCH_SIZE = bs
                LEARNING_RATE = lr
                NUM_EPOCHS = 1000
                EMBEDDING_DIM = ed
                OUTPUT_DIM = EMBEDDING_DIM
                NUM_ENTITIES = 114
                NUM_RELATIONS = 4
                resopnse_pairs = pd.read_csv("../data/data_single_ABL1/ABL1_triples.csv", header=0)
                resopnse_pairs = shuffle(resopnse_pairs, random_state=24)
                mu_similar_triples = pd.read_csv(f"../data/data_single_ABL1/mu_similar0.75.csv", header=0)

                drug_similar_triples = pd.read_csv(f"../data/data_single_ABL1/drug_similar0.75.csv", header=0)

                X_train_neg = pd.read_csv('../data/data_single_ABL1/negative_dc_ABL1.csv', header=0)


                resopnse_pairs.columns = ['obj', 'rel', 'sbj']
                mu_similar_triples.columns = ['obj', 'rel', 'sbj']
                drug_similar_triples.columns = ['obj', 'rel', 'sbj']

                num_splits = 5

                for mode in range(0,1):

                    train_test_splits = utils1.split_pos_triple_into_folds(resopnse_pairs,mu_similar_triples,drug_similar_triples, num_folds=num_splits, seed=SEED,mode=mode)
                    # train_test_splits = utils.split_neg_triple_into_folds(resopnse_pairs, num_folds=num_splits, seed=SEED,mode=mode)
                    neg_train_test_splits = utils1.split_neg_triple_into_folds(X_train_neg, num_folds=num_splits, seed=SEED,mode=mode)

                    for fold in range(0,5):

                        X_train_response, X_test_response = train_test_splits[fold]
                        neg_X_train, neg_X_test = neg_train_test_splits[fold]
                        neg_X_test_filtered = neg_X_test[neg_X_test['rel'].isin([0, 1])]
                        neg_X_test_filtered.to_csv(f'../data/data_single_ABL1/mode{mode}_fold{fold}_neg_X_test.csv',index_label=None)



                        X_train_triple = X_train_response
                        X_test_triple = X_test_response
                        X_test_triple.drop(X_test_triple[(X_test_triple['rel'] == 2) | (X_test_triple['rel'] == 3)].index, inplace=True)
                        syn_X_train_triple = pd.DataFrame(utils1.generate_reverse_triplets(X_train_triple.to_numpy()))
                        syn_neg_X_train = pd.DataFrame(utils1.generate_reverse_triplets(neg_X_train.to_numpy()))
                        syn_X_test_triple = pd.DataFrame(utils1.generate_reverse_triplets(X_test_triple.to_numpy()))
                        syn_neg_X_test = pd.DataFrame(utils1.generate_reverse_triplets(neg_X_test.to_numpy()))

                        syn_X_train_triple.columns = ['obj', 'rel', 'sbj'];
                        syn_X_test_triple.columns = ['obj', 'rel', 'sbj']
                        syn_neg_X_train.columns = ['obj', 'rel', 'sbj'];
                        syn_neg_X_test.columns=['obj', 'rel', 'sbj']

                        neg_X_train = pd.concat([neg_X_train,syn_neg_X_train],axis=0)
                        neg_X_test = pd.concat([neg_X_test,syn_neg_X_test],axis=0)

                        X_train = pd.concat([X_train_triple, syn_X_train_triple], axis=0).astype(np.int64)
                        X_test = pd.concat([X_test_triple, syn_X_test_triple],axis=0).astype(np.int64)
                        # X_test = X_test_triple.astype(np.int64)

                        X_train.to_csv(f"../data/data_single_ABL1/mode{mode}_fold{fold}_X_train.csv", index=False)
                        X_test.to_csv(f"../data/data_single_ABL1/mode{mode}_fold{fold}_X_test.csv", index=False)

                        # ------------------------------------------------------node_representation
                        all_feature_matrix = pd.read_csv(f"../data/data_single_ABL1/feature_ABL1_248.csv", header=None,index_col=0)

                        ADJ_MATS = utils1.get_adj_mats(X_train.values, NUM_ENTITIES, NUM_RELATIONS)

                        X_train = np.expand_dims(X_train, axis=0)

                        X_train_neg = np.expand_dims(neg_X_train, axis=0)
                        np.save(f'../data/data_single_ABL1/mode{mode}_fold{fold}_X_train_neg.npy', X_train_neg)

                        ALL_INDICES = np.arange(NUM_ENTITIES).reshape(1, -1)

                        model = get_IDDGCN_Model(
                            num_entities=NUM_ENTITIES,
                            num_relations=NUM_RELATIONS,
                            embedding_dim=EMBEDDING_DIM,
                            output_dim=OUTPUT_DIM,
                            seed=SEED,
                            all_feature_matrix=all_feature_matrix,
                            mode=mode,
                            fold=fold
                        )
                        model.reset_states()

                        model.compile(
                            loss=tf.keras.losses.BinaryCrossentropy(),
                            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                        )


                        save_path_template = os.path.join('..', 'data','data_single_ABL1', 'weights','mode{mode}_fold{fold}_epoch{epoch}_learnRate{learning_rate}_batchsize{batch_size}_embdim{EMBEDDING_DIM}_weight.h5')

                        save_weights_callback = SaveWeightsCallback(save_epochs=save_epochs, save_path_template=save_path_template, mode=mode, fold=fold, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, EMBEDDING_DIM=EMBEDDING_DIM)

                        history = model.fit(
                            x=[
                                ALL_INDICES,
                                X_train[:, :, 0],
                                X_train[:, :, 1],
                                X_train[:, :, 2],
                                ADJ_MATS
                            ],
                            y=np.ones(X_train.shape[1]).reshape(1, -1),
                            epochs=NUM_EPOCHS,
                            batch_size=BATCH_SIZE,
                            verbose=1,
                            callbacks=[save_weights_callback]
                        )

                        print('len(X_train_response)',len(X_train_response))
                        print('len(X_train),len(X_test)', len(X_train[0]), len(X_test))
                        print(f'len(neg_X_train),len(neg_X_test): ',len(neg_X_train),len(neg_X_test))
                        print(f'Done mode{mode}_fold{fold}')
