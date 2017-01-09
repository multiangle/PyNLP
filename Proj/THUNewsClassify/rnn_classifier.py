import tensorflow as tf
import numpy as np
import math
import pickle as pkl
import jieba
import os
import TextDeal
from Proj.THUNewsClassify.util import pick_valid_word,pick_valid_word_chisquare,read_text,rm_words

class RNNClassifier():
    def __init__(self,
                 label_size,
                 max_depth,
                 batch_size = 1,
                 embed_size=200,
                 hidden_size=400):
        self.embed_size = embed_size
        self.max_depth = max_depth
        self.batch_size = batch_size
        self.label_size = label_size
        self.hidden_size = hidden_size

        self.buildGraph()
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init_op)

    def buildGraph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputs = tf.placeholder(tf.float32,[self.batch_size,self.max_depth,self.embed_size]) # inputs: num_step * embed_size
            self.seq_len = tf.placeholder(tf.int32)
            self.label = tf.placeholder(tf.int32,[1])
            label_float = tf.cast(self.label_size,tf.float32)

            label_matrix = tf.diag(tf.ones(self.label_size))
            embed_label = tf.nn.embedding_lookup(label_matrix,self.label)

            # input_list = list(tf.split(0,self.max_depth,expand_inputs))
            input_list = tf.unpack(self.inputs,axis=1)    # [[1,embed_size,]...,[1,embed_size]]
            # BasicRNNCell: [num_units, input_size, ...]
            self.rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.hidden_size,self.embed_size)
            self.rnn_cell = tf.nn.rnn_cell.DropoutWrapper(self.rnn_cell,output_keep_prob=0.9)
            init_stat = self.rnn_cell.zero_state(1,tf.float32)
            output_embedding,states = tf.nn.rnn(self.rnn_cell,input_list,
                                                initial_state=init_stat,
                                                sequence_length=self.seq_len)
            final_output = output_embedding[-1] # final_output : [1,hidden_size]

            weight = tf.Variable(tf.truncated_normal([self.label_size,self.hidden_size],
                                                     stddev=1.0/math.sqrt(self.hidden_size)))
            biase = tf.Variable(tf.zeros([self.label_size]))

            tmp_y = tf.matmul(final_output,weight,transpose_b=True) + biase
            tmp_g = tf.sigmoid(tmp_y)

            self.predict = tf.cast(tf.argmax(tmp_g,axis=1),tf.float32)
            self.error_num = tf.count_nonzero(label_float-self.predict)

            tiny_v = 0.0001
            self.loss =  tf.reduce_mean(embed_label*tf.log(tmp_g+tiny_v) + (1-embed_label)*tf.log(1+tiny_v-tmp_g))

            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=1).minimize(self.loss)
            self.init_op = tf.global_variables_initializer()

if __name__=='__main__':
    with open('word_list_path.pkl','rb') as f:
        word_info_list = pkl.load(f)
        word2id,id2word = pick_valid_word_chisquare(word_info_list,30000)
    with open('THUCNews.pkl','rb') as f:
        embedding = pkl.load(f)
    with open('file_info_list.pkl','rb') as f:
        file_info_list = pkl.load(f)
    label_list = []
    for info in file_info_list:
        label = info['label']
        if label not in label_list:
            label_list.append(label)

    print(label_list)
    label_size = label_list.__len__()
    embed_size = embedding[0].__len__()
    valid_contexts = []
    valid_label_ids = []
    valid_seq_len = []

    # 遍历一遍，筛选出有效词，以id形式保存，并记录每篇文档中单词数目
    id_type_path = 'THUNews_id_type.pkl'
    if os.path.exists(id_type_path):
        with open(id_type_path,'rb') as f:
            v = pkl.load(f)
            valid_contexts,valid_label_ids,valid_seq_len,_ = v[:]
    else:
        for i,file_info in enumerate(file_info_list):
            if i%1000==0:
                print('已经分词',i,'篇文章')
            file_path = file_info['path']
            file_label = file_info['label']
            lines = read_text(file_path)
            context = "".join(lines)
            words = jieba.cut(context,cut_all=False)
            valid_words_ids = []
            for word in words:
                if (word in word2id) and (not TextDeal.isStopWord(word)):
                    word_id = word2id[word]
                    valid_words_ids.append(word_id)
            if len(valid_words_ids)==0:
                continue
            valid_seq_len.append(len(valid_words_ids))
            valid_contexts.append(valid_words_ids)
            label_id = label_list.index(file_label)
            valid_label_ids.append(label_id)
        with open(id_type_path,'wb') as f:
            v = [valid_contexts,valid_label_ids,valid_seq_len,'从0到2分别是：id模式的所有文章数据,label,每篇文章的有效单词数']
            pkl.dump(v,f)

    # # 对每个文本生成长为max_depth的embed序列，多余部分补零,丢入rnn模型
    # max_depth = max(valid_seq_len)
    # for i,context in enumerate(valid_contexts):
    #      print(context)