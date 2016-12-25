
import tensorflow as tf
import numpy as np
import pickle as pkl
import os.path as path
import os
import math
from pprint import pprint
import collections

class SimpleClassifier():
    def __init__(self,
                 word_list=None,
                 embedding=None,
                 tag_num=None,
                 batch_size=10,
                 load_from=None,
                 ):
        # 如果是新建模型，则 word_list 和 embedding 不能为空
        if load_from!=None:  # 如果导入之前的模型
            self.load_model(load_from)

        self.word_list = word_list  # e.g. ['我','你','他们']
        self.embedding = embedding
        self.tag_num = tag_num
        self.batch_size = batch_size    # 一批中数据个数
        assert self.word_list.__len__() == self.embedding.__len__()

        self.dict_size = self.word_list.__len__() # 词典的规模
        self.embed_size = embedding.shape[1]      # embedding的长度

        self.word2id = {}   # word -> id 映射
        for i in range(self.dict_size):
            self.word2id[word_list[i]] = i

        # 辅助参数
        self.train_sent_num = 0     # 训练的句子数
        self.train_batch_num = 0    # 训练的批次数目
        self.train_loss_records = collections.deque(maxlen=10)  # loss 记录
        self.train_accu_records = collections.deque(maxlen=20)  # 分类正确率记录

        # 构建模型，参数初始化
        self.build_graph(self.embedding)
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)
        self.summary_writer = tf.train.SummaryWriter('/tmp/w2v_simple_classifier',self.sess.graph)

    def build_graph(self,embedding):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.batch_embed_input = tf.placeholder(tf.float32, shape=[self.batch_size, self.embed_size])
            self.batch_label = tf.placeholder(tf.int32,shape=[self.batch_size])
            batch_size = self.batch_label.get_shape()[0]
            self.var_w = tf.Variable(
                tf.truncated_normal([self.tag_num,self.embed_size],
                                    stddev=1.0/math.sqrt(self.embed_size)),
                dtype=tf.float32
            )

            self.var_bias = tf.Variable(tf.zeros([self.tag_num]),dtype=tf.float32)

            # batch_embed_input : [batch_size, embed_size]
            # var_w : [tag_num, embed_size]
            # logits: [batch_size, tag_num]
            logits = tf.matmul(self.batch_embed_input, self.var_w, transpose_b=True) + self.var_bias

            self.predict = tf.expand_dims(tf.to_int32(tf.argmax(logits,axis=1)),0)
            labels = tf.expand_dims(self.batch_label,0)

            self.accuracy = tf.matmul(self.predict,labels,transpose_b=True)

            self.loss = tf.nn.seq2seq.sequence_loss([logits],
                                               [self.batch_label],
                                               [tf.ones([batch_size])])

            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(self.loss)

            self.init = tf.global_variables_initializer()

            tf.scalar_summary('loss',self.loss)
            tf.scalar_summary('accuracy',self.accuracy)
            self.merge_summary_op = tf.merge_all_summaries()

    def train_by_sentences(self,sentences,labels):
        # sentences: 其实是一个单词列表，例如 [['叙利亚','战士',‘解放’],['谷歌','发布','产品'],...]
        # labels: 一个list,例如[2,1,3,1,5]
        assert type(sentences)==list
        assert type(labels)==list
        assert sentences.__len__() == labels.__len__()
        sent_num = sentences.__len__()
        batch_num = sent_num // self.batch_size

        for i in range(batch_num):
            # 处理第 i 批数据
            start_index = i*self.batch_size
            end_index = (i+1)*self.batch_size
            batch_sentence_embed = []
            batch_label = []
            for index in range(start_index,end_index):
                if labels[index]>=self.tag_num:
                    raise RuntimeError("类别id与类别数目不符")
                batch_label.append(labels[index])
                line = sentences[index]
                sentence_embed = None
                for word in line:
                    id = self.word2id.get(word,default=None)
                    if id!=None:
                        if sentence_embed==None:
                            sentence_embed = self.embedding[id,:]
                        else:
                            sentence_embed += self.embedding[id,:]
                batch_sentence_embed.append(sentence_embed)
            batch_input = np.array(batch_sentence_embed)
            batch_label = np.array(batch_label)

            feed_dict = {self.batch_embed_input:batch_input, self.batch_label:batch_label}
            _, loss, accuracy, summary_str = self.sess.run([self.train_op,
                                                            self.loss,
                                                            self.accuracy,
                                                            self.merge_summary_op],
                                                           feed_dict=feed_dict)
            self.train_loss_records.append(loss)
            self.train_accu_records.append(accuracy)
            if self.train_sent_num%100 == 0:
                avg_loss = sum(self.train_loss_records) / self.train_loss_records.__len__()
                avg_accu = sum(self.train_accu_records) / self.train_accu_records.__len__()
                self.summary_writer.add_summary(summary_str, self.train_sent_num)
                print("{a} sentences dealed, loss: {b}, accuracy: {c}"
                      .format(a=self.train_sent_num, b=avg_loss, c=avg_accu))
            self.train_sent_num += self.batch_size
            self.train_batch_num += 1


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

f = open('../embedding/word_info_list.pkl','rb')
word_info_list = pkl.load(f)
word_list = [x['word'] for x in word_info_list]
embedding = [x['embedding'] for x in word_info_list]
embedding_dict = np.array(embedding)
classifier = SimpleClassifier(word_list=word_list,
                              embedding = embedding_dict,
                              tag_num = 5)
