
import tensorflow as tf
import numpy as np
import pickle as pkl
import os.path as path
import os
import math

class SimpleClassifier():
    def __init__(self, word_list=None, embedding=None, tag_num=None, load_from=None):
        # 如果是新建模型，则 word_list 和 embedding 不能为空
        if load_from!=None:  # 如果导入之前的模型
            self.load_model(load_from)

        self.word_list = word_list  # e.g. ['我','你','他们']
        self.embedding = embedding
        self.tag_num = tag_num
        assert self.word_list.__len__() == self.embedding.__len__()

        self.dict_size = self.word_list.__len__() # 词典的规模
        self.embed_size = embedding.shape[1]      # embedding的长度

        self.word2id = {}   # word -> id 映射
        for i in range(self.dict_size):
            self.word2id[word_list[i]] = i

    def build_graph(self,embedding):
        batch_embed_input = tf.placeholder(tf.int32, shape=[None, self.embed_size])
        batch_label = tf.placeholder(tf.int32,shape=[None])

        var_w = tf.Variable(
            tf.truncated_normal([self.tag_num,self.embed_size],
                                stddev=1.0/math.sqrt(self.embed_size))
        )

        var_bias = tf.Variable(tf.zeros([self.tag_num]))

        



    def load_model(self,model_path):
        # todo 未完成，根据 save_model来写
        f = open(model_path,'rb')
        md = pkl.load(f)

    def save_model(self,save_path):
        model_info = {}
        var_names = ['word_list',  # list
                     'embedding',  # np.ndarray
                     'dict_size',  # int
                     'word2id',    # dict
                     'tag_list']
        for var in var_names:
            model_info[var] = eval('self.'+var)
        if path.exists(save_path):
            os.remove(save_path)
        with open(save_path,'wb') as f:
            pkl.dump(model_info,f)



