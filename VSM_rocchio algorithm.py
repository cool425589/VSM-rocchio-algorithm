from __future__ import division
import os
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity as cs
import math
import sys
#from sklearn import cluster, datasets, metrics
#data path
path ='C:/'
basepath = os.path.join(os.path.sep, path ,'Users','M10615079','Desktop','Homework4')
os.chdir (basepath)
word_size = 51253
document_size = 2265
query_size = 800
d_T_dic = []
q_T_dic = []
idf_dic = []
count_d_T_dic = []
pre_result = []
subdocument_name = []
subquery_name = []
a = 0.275
b = 0.725
k_top = 3
pre_k_result = np.zeros( (query_size, k_top) )
pre_k_result.fill(-1)
pre_k_result_size = 10
"""
==============================================================Universal function========================================
"""
def progressbar(cur, total):
    percent = '{:.2%}'.format(cur / total)
    sys.stdout.write('\r')
    sys.stdout.write("[%-50s] %s" % (
                            '=' * int(math.floor(cur * 50 / total)),
                            percent))
    sys.stdout.flush()
def Get_subfile_list(filename):
    os.chdir(filename)
    subfile_list = os.listdir(os.getcwd())
    os.chdir("..")
    return subfile_list
def Get_read_data_func(fileName):
    def read_document_data(subfileName):
        read_path = os.path.join(os.path.sep,os.getcwd(),fileName,subfileName)
        assert os.path.isfile(read_path) , read_path + "not exists"
        with open(read_path, 'r') as f:
            return np.asarray(re.findall('(?<=\s)\d+(?=\s)|^\d+(?=\s)|(?<=\s)\d+$', f.read())).astype(np.integer)         
    return np.frompyfunc(read_document_data, 1, 1)
def init():
    global d_T_dic, q_T_dic, idf_dic, count_d_T_dic, pre_result
    d_T_dic = np.zeros((document_size, word_size))  
    q_T_dic = np.zeros((query_size, word_size))
    idf_dic = np.zeros((word_size))
    idf_dic.fill(-1)
    count_d_T_dic = np.zeros((document_size, word_size))
    pre_result =  np.ndarray( (query_size,k_top) )
def init_subfilename():
    subdocument_name[:] = Get_subfile_list('Document')
    subquery_name[:] = Get_subfile_list('Query')    
def init_count_func(Name_tf_dic):
    def update_tf(read_data, document_number):
        counts = np.bincount(read_data,  minlength = word_size)
        Name_tf_dic[document_number, :] = counts
    return np.frompyfunc(update_tf, 2, 0)
def init_tf_by_FolderName(FolderName, Name_tf_dic):
    update_tf_func(Name_tf_dic)(Get_read_data_func(FolderName)(Get_subfile_list(FolderName)), np.arange(len(Get_subfile_list(FolderName))))
def init_count_by_FolderName(FolderName, Name_tf_dic):
    init_count_func(Name_tf_dic)(Get_read_data_func(FolderName)(Get_subfile_list(FolderName)), np.arange(len(Get_subfile_list(FolderName))))
def write_result_to_text(write_file, query_number, Retrievalarray):
    write_file.write('\n' + subquery_name[query_number] + ',')
    for docnum in Retrievalarray[0:100]:
        write_file.write(subdocument_name[docnum]+' ')
    save_pre_result(query_number, Retrievalarray)
def save_pre_result(query_number, Retrievalarray):
    num = 0
    for docnum in Retrievalarray[0:k_top]:
            pre_result[query_number, num] = docnum
            num += 1
"""
=============================================================Update idf====================================================
"""
def update_tf_func(Name_tf_dic):
    def update_tf(read_data, document_number):
        counts = np.bincount(read_data,  minlength = word_size)
        with np.errstate(divide='ignore'):     
            Name_tf_dic[document_number, :] = np.where(counts==0, 0, counts/np.sum(counts))
        progressbar(document_number+1, Name_tf_dic.shape[0])
    return np.frompyfunc(update_tf, 2, 0)
def Get_idf(term):
    sum = np.sum(np.count_nonzero(d_T_dic[:,term]))
    return np.log( document_size / (1 + sum) )
def update_idf(term, idf):
    idf_dic[term] = idf
def Get_idf_from_dic(term):
    idf  = Get_idf(term)
    update_idf(term, idf)
    return idf
def update_idf_func():
    def idf_from_dic(term):
        Get_idf_from_dic(term)
        progressbar(term+1, word_size)
    return np.frompyfunc(idf_from_dic, 1, 0)  
def init_idf():
    update_idf_func()(np.arange(word_size))
"""
=============================================================Similarity====================================================
"""
def cosine_similarity(Q, D):
    rank_result = cs(Q, D)  
    return np.argsort(-rank_result)
def rank_ALL_func(write_file, result_matrix):
    def rank_ALL(query_number):   
        write_result_to_text(write_file, query_number, result_matrix[query_number,:])
        progressbar(query_number+1, query_size)
    return np.frompyfunc(rank_ALL, 1, 0)
"""
=============================================================Rocchio Algorithm ====================================================
"""
def Ger_new_query(querynumber):
    index_array = pre_result[querynumber, :].astype(np.integer)
    return a * q_T_dic[querynumber, :] + b * np.sum( count_d_T_dic[ index_array, : ] ,axis = 0) / k_top
def Update_new_query():
    for querynumber in np.arange(query_size):
        q_T_dic[querynumber, :] = Ger_new_query(querynumber)
        progressbar(querynumber+1, query_size)
"""
===========================================================K-means culstering======================================================
"""
#input pre_result[100]
def Get_best_k_in_K_means(querynum, feature_matrix, min_ks, max_ks):
    best_k = 0
    best_score = 0
    ks = range(min_ks, max_ks)
    global pre_k_result
    cluster_label = []
    transform = []
    for k in ks:
        kmeans_fit = cluster.KMeans(n_clusters = k, precompute_distances = True).fit(feature_matrix)
        local_cluster_labels = kmeans_fit.labels_
        local_transform = kmeans_fit.transform(feature_matrix)
        score = metrics.silhouette_score(feature_matrix, local_cluster_labels)
        if (score > best_score):
            best_score = score
            best_k = k
            cluster_label = local_cluster_labels    
            transform = local_transform
    
    distense = np.sum( transform, axis = 1)
    for k in range(best_k):
        document_num, = np.where(cluster_label == k)
        center_num = document_num[ np.argmin(distense[document_num]) ]
        pre_k_result[querynum, k] = center_num
def Ger_new_query_kmeans(querynumber):
    index_array = pre_result[querynumber, :].astype(np.integer)
    index_array = index_array[0:pre_k_result_size]
    Get_best_k_in_K_means(querynumber, d_T_dic[ index_array, : ], 2, 3)
    Get_K_means_index = pre_k_result[querynumber, :]
    Get_K_means_index = Get_K_means_index[Get_K_means_index >=0]
    index_array = index_array[ np.array(Get_K_means_index).astype(np.integer) ]
    return a * q_T_dic[querynumber, :] + b * np.sum( count_d_T_dic[ index_array, : ] ,axis = 0) / len(index_array)
def Update_new_query_kmeans():
    pre_k_result.fill(-1)
    for querynumber in np.arange(query_size):
        q_T_dic[querynumber, :] = Ger_new_query_kmeans(querynumber)
        progressbar(querynumber+1, query_size)
"""
=============================================================main====================================================
"""
def main():
    init()
    init_subfilename()  
    init_count_by_FolderName('Document', count_d_T_dic)
    print ("count Document tf")
    init_tf_by_FolderName('Document', d_T_dic)
    print ("\ncount Query tf")
    init_tf_by_FolderName('Query', q_T_dic)
    print ("\ncount word idf")
    init_idf()         
    write_result_path = os.path.join(os.path.sep,os.getcwd(),'result.txt')
    with open(write_result_path, 'w', encoding = 'UTF-8') as write_file :
        write_file.write('Query,RetrievedDocuments')
        print("\nfirst start ranking")
        rank_ALL_func(write_file, cosine_similarity(q_T_dic, d_T_dic*idf_dic[np.newaxis,:]) )(np.arange(query_size))
        write_file.close()    
    for num in range(10):       
        write_result_path = os.path.join(os.path.sep,os.getcwd(),'new_query_result_'+str(num)+'_k'+str(k_top)+'.txt')
        print("\nUpdate Rocchio Algorithm query , Step" + str(num+1))
        Update_new_query()
        with open(write_result_path, 'w', encoding = 'UTF-8') as write_file :
            write_file.write('Query,RetrievedDocuments')
            print("\nRocchio Algorithm start ranking , Step" + str(num+1))
            rank_ALL_func(write_file, cosine_similarity(q_T_dic*idf_dic[np.newaxis,:], d_T_dic*idf_dic[np.newaxis,:]))(np.arange(query_size))
            write_file.close()                                                                                           
main()





