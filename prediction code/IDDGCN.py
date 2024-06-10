#!/usr/bin/env python3
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.layers import Embedding, Lambda
import utils1
import pandas as pd
import os
import numpy as np
import random as rn
from datetime import datetime
from keras.callbacks import Callback,EarlyStopping
from sklearn.decomposition import PCA
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class IDDGCN_Layer(tf.keras.layers.Layer):
    def __init__(self, num_entities, num_relations, output_dim, seed, **kwargs):
        super(IDDGCN_Layer, self).__init__(**kwargs)
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.output_dim = output_dim
        self.seed = seed

        # 创建关系和自连接的权重矩阵
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

        # 添加关系权重参数
        self.relation_weights = self.add_weight(
            shape=(self.num_relations,),
            initializer='uniform',
            trainable=True,
            name='relation_weights'
        )

        # 添加动态关系权重计算的权重和偏置
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
        # self.alpha_values = None
    def call(self, inputs):
        embeddings, head_idx, head_e, tail_idx, tail_e, *adj_mats = inputs
        head_output = tf.matmul(head_e, self.self_kernel)
        tail_output = tf.matmul(tail_e, self.self_kernel)

        # 计算每种关系的动态权重
        alpha = tf.nn.softmax(tf.matmul(head_e, self.W_alpha) + self.b_alpha)
        # self.alpha_values = alpha  # 保存alpha值
        for i in range(self.num_relations):
            adj_i = tf.sparse.reshape(adj_mats[0][i], shape=(self.num_entities, self.num_entities))
            sum_embeddings = tf.sparse.sparse_dense_matmul(adj_i, embeddings)
            head_update = tf.nn.embedding_lookup(sum_embeddings, head_idx)
            tail_update = tf.nn.embedding_lookup(sum_embeddings, tail_idx)

            # 应用动态关系权重
            relation_weight = tf.sigmoid(alpha[:, i])
            head_output += tf.expand_dims(relation_weight, 1) * tf.matmul(head_update, self.relation_kernel[i])
            tail_output += tf.expand_dims(relation_weight, 1) * tf.matmul(tail_update, self.relation_kernel[i])

        return tf.sigmoid(head_output), tf.sigmoid(tail_output)


class DistMult(tf.keras.layers.Layer):
    # 计算知识图谱中的关系得分
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

        score = tf.sigmoid(tf.reduce_sum(head_e * rel_e * tail_e, axis=-1))  # 对同一行的元素进行累加     这里计算返回的分数
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
        # data包含了输入数据和相应的标签，即x和y
        all_indices, pos_head, rel, pos_tail, *adj_mats = data[0]
        y_pos_true = data[1]
        # 使用 pandas 读取 CSV 文件
        X_train_neg = np.load(f'../../data/mode{self.mode}_fold{self.fold}_X_train_neg.npy')
        # 获取数据集的长度（样本数）
        num_samples = X_train_neg.shape[1]

        # 生成随机索引
        random_indices = np.random.permutation(num_samples)

        # 使用随机索引重新排列数据集
        X_train_neg = X_train_neg[:, random_indices, :]
        neg_head = X_train_neg[:,:,0]
        neg_tail = X_train_neg[:,:,2]
        neg_rel = X_train_neg[:,:,1]
        # 转换为张量
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


            # 计算损失关于模型参数的梯度
            grads = tape.gradient(loss, self.trainable_weights)
            # 用梯度更新模型参数
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            # 更新模型的度量状态
            self.compiled_metrics.update_state(y_pos_true, y_pos_pred)
            # 返回模型的度量
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
            # 保存模型权重
            filename = self.save_path_template.format(mode=self.mode, fold=self.fold, epoch=epoch + 1,
                                                      learning_rate=self.learning_rate, batch_size=self.batch_size,
                                                      EMBEDDING_DIM=self.EMBEDDING_DIM)
            self.model.save_weights(filename)
            print(f"\nSaved weights for epoch {epoch + 1} to {filename}")

def get_IDDGCN_Model(num_entities,num_relations, embedding_dim, output_dim, seed, all_feature_matrix,mode,fold):
    # 输入层
    head_input = tf.keras.Input(shape=(None,), name='head_input', dtype=tf.int64)
    rel_input = tf.keras.Input(shape=(None,), name='rel_input', dtype=tf.int64)
    tail_input = tf.keras.Input(shape=(None,), name='tail_input', dtype=tf.int64)
    all_entities = tf.keras.Input(shape=(None,), name='all_entities', dtype=tf.int64)

    adj_inputs = [tf.keras.Input(
        shape=(num_entities, num_entities),
        dtype=tf.float32,
        name='adj_inputs_' + str(i),
        sparse=True,
    ) for i in range(num_relations)]  # 生成的同样是列表，每种关系一种邻接矩阵

    # 嵌入层
    entity_embeddings = Embedding(
        input_dim=num_entities,
        output_dim=embedding_dim,
        name='entity_embeddings',
    #     # 嵌入矩阵的初始值设置
    #     # weights=[tf.constant(all_feature_matrix, dtype=tf.float32)],
        trainable=True,
        embeddings_initializer=tf.keras.initializers.RandomUniform(
             minval=0,
             maxval=1,
             seed=seed)
    #     #自动生成特征-==============================================================
        )
    # # //////////////////////////////////////////////////
    # pretrained_weights = all_feature_matrix.astype(np.float32)
    # pca = PCA(n_components=64)
    # pretrained_weights = pca.fit_transform(pretrained_weights)
    # pretrained_weights = (pretrained_weights - np.mean(pretrained_weights, axis=0)) / np.std(pretrained_weights, axis=0)
    # entity_embeddings = tf.keras.layers.Embedding(
    #     input_dim=num_entities,
    #     output_dim=embedding_dim,
    #     name='entity_embeddings',
    #     embeddings_initializer=tf.keras.initializers.Constant(pretrained_weights),
    #     trainable=True
    # )
    head_e = entity_embeddings(head_input)  # 生成嵌入层
    tail_e = entity_embeddings(tail_input)
    all_e = entity_embeddings(all_entities)
    # 匿名函数层
    head_e = Lambda(lambda x: x[0, :, :])(head_e)
    tail_e = Lambda(lambda x: x[0, :, :])(tail_e)
    all_e = Lambda(lambda x: x[0, :, :])(all_e)
    head_index = Lambda(lambda x: x[0, :])(head_input)
    rel_index = Lambda(lambda x: x[0, :])(rel_input)
    tail_index = Lambda(lambda x: x[0, :])(tail_input)

    # 第一层 IDDGCN
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

    # 第二层 IDDGCN
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

    # 第三层 IDDGCN
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

if __name__ == '__main__':

    SEED = 89
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '0'
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    rn.seed(SEED)
    # 网格调参法
    batch_size = [100]
    learning_rate = [0.001]
    embedding_dim = [64]
    save_epochs = [1,100,500,1000,2000,3000,5000]

    for bs in batch_size:
        for lr in learning_rate:
            for ed in embedding_dim:

                BATCH_SIZE = bs
                LEARNING_RATE = lr
                NUM_EPOCHS = 5000
                EMBEDDING_DIM = ed
                OUTPUT_DIM = EMBEDDING_DIM
                NUM_ENTITIES = 845
                NUM_RELATIONS = 4
                # allpairs,resopnse_pairs,cell_similar_triples,drug_similar_triples,cell_entities_map,drug_entities_map = data_process(EMBEDDING_DIM,SEED)
                resopnse_pairs = pd.read_csv("../data/triplets_dc.csv", header=0)
                resopnse_pairs = shuffle(resopnse_pairs, random_state=24)
                cell_similar_triples = pd.read_csv(f"../data/mu_similar0.97.csv", header=0)

                drug_similar_triples = pd.read_csv(f"../data/drug_similar0.78.csv", header=0)

                X_train_neg = pd.read_csv('../data/negative_dc_28_1754.csv', header=0)
                # ----------------------------------------------------保存三元组，以实体形式

                resopnse_pairs.columns = ['obj', 'rel', 'sbj']
                cell_similar_triples.columns = ['obj', 'rel', 'sbj']
                drug_similar_triples.columns = ['obj', 'rel', 'sbj']

                num_splits = 5
                # 0是正常实验，1是no_cell，2是no_drug，3是no_cell_drug
                for mode in range(0,1):

                    train_test_splits = utils1.split_pos_triple_into_folds(resopnse_pairs,cell_similar_triples,drug_similar_triples, num_folds=num_splits, seed=SEED,mode=mode)
                    # train_test_splits = utils1.split_neg_triple_into_folds(resopnse_pairs, num_folds=num_splits, seed=SEED,mode=mode)
                    neg_train_test_splits = utils1.split_neg_triple_into_folds(X_train_neg, num_folds=num_splits, seed=SEED,mode=mode)

                    for fold in range(4,5):

                        X_train_response, X_test_response = train_test_splits[fold]
                        neg_X_train, neg_X_test = neg_train_test_splits[fold]
                        neg_X_test_filtered = neg_X_test[neg_X_test['rel'].isin([0, 1])]
                        neg_X_test_filtered.to_csv(f'../data/mode{mode}_fold{fold}_neg_X_test.csv',index_label=None)
                        # 获取X_train和X_test

                        # X_train_triple = pd.concat([X_train_response, cell_similar_triples, drug_similar_triples], axis=0)
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

                        X_train.to_csv(f"../data/mode{mode}_fold{fold}_X_train.csv", index=False)
                        X_test.to_csv(f"../data/mode{mode}_fold{fold}_X_test.csv", index=False)

                        # ------------------------------------------------------提取node_representation
                        all_feature_matrix = pd.read_csv(f"../data/feature_all_248.csv", header=None,index_col=0)
                        # all_feature_matrix = all_feature_matrix.apply(lambda x: np.maximum(0, x))
                        # all_feature_matrix = 1 / (1 + np.exp(-all_feature_matrix))
                        #...........................激活函数
                        # 生成三元组对应的矩阵
                        ADJ_MATS = utils1.get_adj_mats(X_train.values, NUM_ENTITIES, NUM_RELATIONS)

                        X_train = np.expand_dims(X_train, axis=0)

                        X_train_neg = np.expand_dims(neg_X_train, axis=0)
                        np.save(f'../data/mode{mode}_fold{fold}_X_train_neg.npy', X_train_neg)

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
                        # 在每个 fold 的循环迭代之前清零模型权重
                        model.reset_states()

                        model.compile(
                            loss=tf.keras.losses.BinaryCrossentropy(),
                            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                        )

                        # 定义保存权重的文件名模板
                        save_path_template = os.path.join('..', 'data', 'weights', 'IDDGCN_normal','mode{mode}_fold{fold}_epoch{epoch}_learnRate{learning_rate}_batchsize{batch_size}_embdim{EMBEDDING_DIM}_weight.h5')
                        # 创建自定义回调函数
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
                        entity_embeddings = model.get_layer('entity_embeddings').get_weights()[0]
                        np.save(f'../data/mode{mode}_fold{fold}_learning_rate0.001_final_embeddings.npy', entity_embeddings)
                        # filename = f'mode{mode}_fold{fold},{num_splits}_epoch{NUM_EPOCHS}_batchsize{BATCH_SIZE}_embdim{EMBEDDING_DIM}.h5'
                        # model.save_weights(os.path.join('..', 'data2', 'weights', 'drug_celline', filename))
                        print('len(X_train_response)',len(X_train_response))
                        print('len(X_train),len(X_test)', len(X_train[0]), len(X_test))
                        print(f'len(neg_X_train),len(neg_X_test): ',len(neg_X_train),len(neg_X_test))
                        print(f'Done mode{mode}_fold{fold}')
