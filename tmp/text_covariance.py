
import numpy as np
import tensorflow as tf
from pymongo import MongoClient
import pickle as pkl
import os
from matplotlib import pyplot as plt

# f = open('../embedding/word_info_list.pkl','rb')
# word_info_list = pkl.load(f)
# embedding_list = [x['embedding'] for x in word_info_list]
# embedding = np.array(embedding_list)
#
# l2_model = np.sqrt(np.sum(np.square(embedding),axis=1))
# l2_model_x = np.expand_dims(l2_model,1)
# tmp = np.ones([1,embedding.shape[1]])
# l2_model_m = np.matmul(l2_model_x,tmp)
#
# # normed_embedding = np.divide(embedding,l2_model_m)
# normed_embedding = embedding / l2_model_m
#
# embedding = normed_embedding
# new_model = np.sqrt(np.sum(np.square(embedding),axis=1))
# new_model.sort()
# plt.plot(new_model,'.')
# plt.show()




f = open('../test/tensorflow/embedding.pkl','rb')
embedding = pkl.load(f)


dim_avg = np.mean(embedding,axis=0)
dim_covar = np.matmul((embedding-dim_avg).T,(embedding-dim_avg)) / embedding.__len__()
eig_val, eig_vec = np.linalg.eig(dim_covar)
v = np.argsort(eig_val)
# eig_val.sort()
print(eig_val)
plt.plot(eig_val)
plt.show()