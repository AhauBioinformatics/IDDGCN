import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
EMBEDDING_DIM = 64
x_mu = pd.read_csv(r"..\data_single_KRAS\mu_KRAS.csv",header=None,index_col=0)
x_drug = pd.read_csv("..\data_single_KRAS\drug_KRAS.csv",header=None,index_col=0)
x_drug=x_drug.dropna()
def caculat_distance(node_feat):

    distance = cosine_similarity(node_feat)
    return distance

def show_mat_elem(mat):

    mat_list = np.reshape(mat, (-1,))
    print(len(mat_list))
    mat_list = np.around(mat_list, decimals=0)
    max_elem = np.amax(mat_list)
    print(list(set(mat_list)))
    print(len(list(set(mat_list))))
    return max_elem
def creat_similar_mat(mat, threshold):
    result_mat = np.zeros_like(mat)
    result_mat[mat > threshold] = 1
    result_mat.astype(int)
    return result_mat




def save_csv(similar_mat,file_name):
    similar_mat = pd.DataFrame(similar_mat)
    similar_mat.to_csv(f'../data_single_KRAS/{file_name}', index=False, header=False)
def simat2triple(mat,relation,start):
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
    bool_mat = (mat == 1)
    count_ones = np.sum(bool_mat)
    similar_edges_num = (count_ones-len(mat))/2
    similar_rate = similar_edges_num/all_edges_num
mu_dist_mat = caculat_distance(x_mu)
drug_dist_mat = caculat_distance(x_drug)
if np.any(mu_dist_mat < 0) and np.any(drug_dist_mat < 0):
    print(" there are negative values in the matrix")
else:
    print("there are no negative values in the matrix")

threshold = 0.75
drug_threshold = 0.75
mu_similar_mat = creat_similar_mat(mu_dist_mat, threshold).astype(int)
drug_similar_mat = creat_similar_mat(drug_dist_mat, drug_threshold).astype(int)

mu_triples = simat2triple(mu_similar_mat,relation=3,start=0)
print(len(x_mu))
drug_triples = simat2triple(drug_similar_mat,relation=2,start=len(x_mu))
print(f'len(mu_triples):{len(mu_triples)},len(drug_triples):{len(drug_triples)}')
column_names = ['obj', 'rel', 'sbj']
save_csv(mu_triples,f'mu_similar{threshold}.csv')
save_csv(drug_triples,f'drug_similar{drug_threshold}.csv')


print('Done')
