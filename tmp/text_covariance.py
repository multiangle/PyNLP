#coding:utf-8
import numpy as np
import tensorflow as tf
from pymongo import MongoClient
import pickle as pkl
import os
from matplotlib import pyplot as plt

f = open('../embedding/word_info_list.pkl','rb')
word_info_list = pkl.load(f)
embedding_list = [x['embedding'] for x in word_info_list]
word_list = [x['word'] for x in word_info_list]
embedding = np.array(embedding_list)

l2_model = np.sqrt(np.sum(np.square(embedding),axis=1))
v_mean = np.mean(embedding,axis=1)
v_mean_total = np.mean(v_mean)
dim_mean = np.mean(embedding,axis=0)
# print(v_mean_total)
embedding = embedding - dim_mean
# plt.plot(np.sum(embedding,axis=0))

# f = open('../test/tensorflow/embedding.pkl','rb')
# embedding = pkl.load(f)

dim_avg = np.mean(embedding,axis=0)
dim_covar = np.matmul((embedding-dim_avg).T,(embedding-dim_avg)) / embedding.__len__()
eig_val, eig_vec = np.linalg.eig(dim_covar)
v = np.argsort(eig_val)
# eig_val.sort()
print(eig_val)
print(sum(eig_val))
plt.plot(eig_val)
plt.show()

# def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
#     assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
#     plt.figure(figsize=(18, 18))  # in inches
#     myfont = fm.FontProperties(fname='/home/multiangle/download/msyh.ttc')
#     for i, label in enumerate(labels):
#         x, y = low_dim_embs[i, :]
#         plt.scatter(x, y)
#         plt.annotate(label,
#                      xy=(x, y),
#                      xytext=(5, 2),
#                      textcoords='offset points',
#                      ha='right',
#                      va='bottom',
#                      fontproperties=myfont)
#
#     plt.savefig(filename)
#
# final_embeddings = embedding
# try:
#     from sklearn.manifold import TSNE
#     import matplotlib.pyplot as plt
#     import matplotlib.font_manager as fm
#     tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
#     plot_only = 1000
#     low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
#     labels = [word_list[i] for i in range(plot_only)]
#     plot_with_labels(low_dim_embs, labels)
#
# except Exception as e:
#     print(e)
#     print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")