import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
EMBEDDING_DIM = 64
x_cell = pd.read_csv(r"..\data_single_KRAS\mu_KRAS.csv",header=None,index_col=0)
x_drug = pd.read_csv("..\data_single_KRAS\drug_KRAS.csv",header=None,index_col=0)
x_drug=x_drug.dropna()
def caculat_distance(node_feat):
    # 计算这些特征之间的余弦相似性
    distance = cosine_similarity(node_feat)
    return distance

def show_mat_elem(mat):
    # 展示保留特定小数后矩阵里的元素有哪些，并返回矩阵最大元素
    mat_list = np.reshape(mat, (-1,))
    print(len(mat_list))
    mat_list = np.around(mat_list, decimals=0)
    max_elem = np.amax(mat_list)
    print(list(set(mat_list)))
    print(len(list(set(mat_list))))
    return max_elem
def creat_similar_mat(mat, threshold):
# 将相似矩阵里的相似元素置为1，其余置为0
    # 创建一个与输入矩阵相同大小的零矩阵
    result_mat = np.zeros_like(mat)
    # 使用 NumPy 条件操作将大于阈值的元素置为1
    result_mat[mat > threshold] = 1
    # 矩阵元素转化为整形
    result_mat.astype(int)
    return result_mat




def save_csv(similar_mat,file_name):
    # 保存矩阵为 CSV 文件
    similar_mat = pd.DataFrame(similar_mat)
    similar_mat.to_csv(f'../data_single_KRAS/{file_name}', index=False, header=False)
def simat2triple(mat,relation,start):
# 相似矩阵转化为三元组
    triples = []
    num_nodes = mat.shape[0]

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i < j and mat[i,j]!= 0:
                i1 = i+start;j1=j+start
                triples.append((i1,relation,j1))
    return triples


def mat_information(mat):
    all_edges_num = (len(mat)**2-len(mat))/2
    # 创建布尔矩阵，1的位置为True，0的位置为False
    bool_mat = (mat == 1)
    # 使用 np.sum() 统计True的数量，即1的数量
    count_ones = np.sum(bool_mat)
    similar_edges_num = (count_ones-len(mat))/2
    similar_rate = similar_edges_num/all_edges_num
    print(f'在{int(all_edges_num)}个边里，共有{similar_edges_num}个相似边,占比{similar_rate:.2f}')
# 计算余弦相似性
cell_dist_mat = caculat_distance(x_cell)
drug_dist_mat = caculat_distance(x_drug)
# 查看相似余弦相似性矩阵中有没有负值
if np.any(cell_dist_mat < 0) and np.any(drug_dist_mat < 0):
    print("矩阵中包含负值")
else:
    print("矩阵中没有负值")

# 给余弦相似性设定阈值，并创建相似性矩阵
threshold = 0.75
drug_threshold = 0.75
cell_similar_mat = creat_similar_mat(cell_dist_mat, threshold).astype(int)
drug_similar_mat = creat_similar_mat(drug_dist_mat, drug_threshold).astype(int)

# -----------------------------------------------------给矩阵添加细胞和药物id索引
# 创建索引和列名的映射
# entity_map = pd.read_csv('../data2/all_maps_df.csv',header=0)
# index_mapping = entity_map.set_index('1')['0'].to_dict()
#
#
# cell_similar_mat = pd.DataFrame(cell_similar_mat)
# drug_similar_mat = pd.DataFrame(drug_similar_mat)
# # 偏移索引从35开始匹配映射值
# drug_similar_mat.index = drug_similar_mat.index + 477
# drug_similar_mat.columns = drug_similar_mat.columns + 477
#
# cell_similar_mat = cell_similar_mat.rename(index=index_mapping, columns=index_mapping)
# drug_similar_mat = drug_similar_mat.rename(index=index_mapping, columns=index_mapping)

# -----------------------------------------------------
# 保存相似矩阵
# save_csv(cell_similar_mat,f'../data2/new_cell_similar_mat{threshold}.csv')
# save_csv(drug_similar_mat,f'../data2/new_drug_similar_mat{drug_threshold}.csv')
# 把相似矩阵转化为三元组并保存
cell_triples = simat2triple(cell_similar_mat,relation=3,start=0)
print(len(x_cell))
drug_triples = simat2triple(drug_similar_mat,relation=2,start=len(x_cell))
print(f'len(cell_triples):{len(cell_triples)},len(drug_triples):{len(drug_triples)}')
print(f'细胞占比：{len(cell_triples)/104/104}，药物占比：{len(drug_triples)/10/10}')
column_names = ['obj', 'rel', 'sbj']
save_csv(cell_triples,f'mu_similar{threshold}.csv')
save_csv(drug_triples,f'drug_similar{drug_threshold}.csv')


print('Done')
# import pandas as pd
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.metrics.pairwise import pairwise_kernels
#
# EMBEDDING_DIM = 64
# x_cell = pd.read_csv("D:\multiple_explanations_main\data\Mutation_feature_248.csv", header=None, index_col=0)
# x_drug = pd.read_csv("D:\multiple_explanations_main\data\Drug_feature_248.csv", header=None, index_col=0)
# x_drug = x_drug.dropna()
#
# def caculat_distance(node_feat):
#     # 计算这些特征之间的余弦相似性
#     distance = cosine_similarity(node_feat)
#     return distance
#
# def calculate_cka_similarity(node_feat):
#     # 计算这些特征之间的CKA相似性
#     distance = pairwise_kernels(node_feat, metric='rbf') / node_feat.shape[0]
#     return distance
#
# def creat_similar_mat(mat, threshold):
#     # 将相似矩阵里的相似元素置为1，其余置为0
#     result_mat = np.zeros_like(mat)
#     result_mat[mat > threshold] = 1
#     result_mat.astype(int)
#     return result_mat
#
# def simat2triple(mat, relation, start):
#     # 相似矩阵转化为三元组
#     triples = []
#     num_nodes = mat.shape[0]
#     for i in range(num_nodes):
#         for j in range(num_nodes):
#             if i < j and mat[i, j] != 0:
#                 i1 = i + start
#                 j1 = j + start
#                 triples.append((i1, relation, j1))
#     return triples
#
# # 计算余弦相似性
# cell_dist_mat = caculat_distance(x_cell)
# # 计算CKA相似性
# drug_dist_mat = calculate_cka_similarity(x_drug)
# def adjust_and_normalize_cka_similarity_matrix(similarity_matrix):
#     # 获取对角线元素
#     diagonal_elements = np.diag(similarity_matrix)
#     # 计算调整因子
#     adjustment_factors = 1 / diagonal_elements
#     # 调整矩阵
#     adjusted_matrix = similarity_matrix * adjustment_factors[:, np.newaxis]
#     adjusted_matrix = adjusted_matrix * adjustment_factors[np.newaxis, :]
#     # 将对角线元素设置为1
#     np.fill_diagonal(adjusted_matrix, 1)
#     # 将非对角线元素限制在0到1的范围内
#     adjusted_matrix[adjusted_matrix < 0] = 0
#     adjusted_matrix[adjusted_matrix > 1] = 1
#     return adjusted_matrix
#
# # 调整并归一化CKA相似性矩阵
# drug_dist_mat = adjust_and_normalize_cka_similarity_matrix(drug_dist_mat)
#
# # 给余弦相似性设定阈值，并创建相似性矩阵
# threshold = 0.9
# drug_threshold = 0.75
# cell_similar_mat = creat_similar_mat(cell_dist_mat, threshold).astype(int)
# drug_similar_mat = creat_similar_mat(drug_dist_mat, drug_threshold).astype(int)
#
# # 把相似矩阵转化为三元组并保存
# cell_triples = simat2triple(cell_similar_mat, relation=3, start=0)
# print(len(x_cell))
# drug_triples = simat2triple(drug_similar_mat, relation=2, start=len(x_cell))
# print(f'len(cell_triples):{len(cell_triples)}, len(drug_triples):{len(drug_triples)}')
# print(f'细胞占比：{len(cell_triples)/661/661}，药物占比：{len(drug_triples)/184/184}')
#
# # 保存相似矩阵
# # save_csv(cell_similar_mat, f'../data2/new_cell_similar_mat{threshold}.csv')
# # save_csv(drug_similar_mat, f'../data2/new_drug_similar_mat{drug_threshold}.csv')
#
# print('Done')