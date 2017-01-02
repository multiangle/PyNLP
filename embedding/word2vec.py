
import tensorflow as tf
import numpy as np
import math
import collections
import pickle as pkl
from pprint import pprint
from pymongo import MongoClient
import re
import jieba
import os.path as path
import os

class NEGModel():
    def __init__(self,
                 vocab_list=None,
                 embedding_size=200,
                 win_len=3, # 单边窗口长
                 num_sampled=1000,
                 learning_rate=1.0,
                 logdir='/tmp/simple_word2vec',
                 model_path= None
                 ):

        # 获得模型的基本参数
        self.batch_size     = None # 一批中数据个数, 目前是根据情况来的
        if model_path!=None:
            self.load_model(model_path)
        else:
            # model parameters
            assert type(vocab_list)==list
            self.vocab_list     = vocab_list
            self.vocab_size     = vocab_list.__len__()
            self.embedding_size = embedding_size
            self.win_len        = win_len
            self.num_sampled    = num_sampled
            self.learning_rate  = learning_rate
            self.logdir         = logdir

            self.word2id = {}   # word => id 的映射
            for i in range(self.vocab_size):
                self.word2id[self.vocab_list[i]] = i

            # train times
            self.train_words_num = 0 # 训练的单词对数
            self.train_sents_num = 0 # 训练的句子数
            self.train_times_num = 0 # 训练的次数（一次可以有多个句子）

            # train loss records
            self.train_loss_records = collections.deque(maxlen=10) # 保存最近10次的误差
            self.train_loss_k10 = 0

        self.build_graph()
        self.init_op()
        if model_path!=None:
            tf_model_path = os.path.join(model_path,'tf_vars')
            self.saver.restore(self.sess,tf_model_path)

    def init_op(self):
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)
        self.summary_writer = tf.train.SummaryWriter(self.logdir, self.sess.graph)

    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            self.embedding_dict = tf.Variable(
                tf.random_uniform([self.vocab_size,self.embedding_size],-1.0,1.0)
            )
            self.nce_weight = tf.Variable(tf.truncated_normal([self.vocab_size, self.embedding_size],
                                                             stddev=1.0/math.sqrt(self.embedding_size)))
            self.nce_biases = tf.Variable(tf.zeros([self.vocab_size]))

            # 将输入序列向量化
            embed = tf.nn.embedding_lookup(self.embedding_dict, self.train_inputs) # batch_size

            # 得到NCE损失
            self.loss = tf.reduce_mean(
                    tf.nn.nce_loss(
                    weights = self.nce_weight,
                    biases = self.nce_biases,
                    labels = self.train_labels,
                    inputs = embed,
                    num_sampled = self.num_sampled,
                    num_classes = self.vocab_size
                )
            )

            # tensorboard 相关
            tf.scalar_summary('loss',self.loss)  # 让tensorflow记录参数

            # 根据 nce loss 来更新梯度和embedding
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(self.loss)  # 训练操作

            # 计算与指定若干单词的相似度
            self.test_word_id = tf.placeholder(tf.int32,shape=[None])
            vec_l2_model = tf.sqrt(  # 求各词向量的L2模
                tf.reduce_sum(tf.square(self.embedding_dict),1,keep_dims=True)
            )

            avg_l2_model = tf.reduce_mean(vec_l2_model)
            tf.scalar_summary('avg_vec_model',avg_l2_model)

            self.normed_embedding = self.embedding_dict / vec_l2_model
            # self.embedding_dict = norm_vec # 对embedding向量正则化
            test_embed = tf.nn.embedding_lookup(self.normed_embedding, self.test_word_id)
            self.similarity = tf.matmul(test_embed, self.normed_embedding, transpose_b=True)

            # 变量初始化
            self.init = tf.global_variables_initializer()

            self.merged_summary_op = tf.merge_all_summaries()

            self.saver = tf.train.Saver()

    def train_by_sentence(self, input_sentence=[]):
        #  input_sentence: [sub_sent1, sub_sent2, ...]
        # 每个sub_sent是一个单词序列，例如['这次','大选','让']
        sent_num = input_sentence.__len__()
        batch_inputs = []
        batch_labels = []
        for sent in input_sentence:
            for i in range(sent.__len__()):
                start = max(0,i-self.win_len)
                end = min(sent.__len__(),i+self.win_len+1)
                for index in range(start,end):
                    if index == i:
                        continue
                    else:
                        input_id = self.word2id.get(sent[i])
                        label_id = self.word2id.get(sent[index])
                        if not (input_id and label_id):
                            continue
                        batch_inputs.append(input_id)
                        batch_labels.append(label_id)
        batch_inputs = np.array(batch_inputs,dtype=np.int32)
        batch_labels = np.array(batch_labels,dtype=np.int32)
        batch_labels = np.reshape(batch_labels,[batch_labels.__len__(),1])

        feed_dict = {
            self.train_inputs: batch_inputs,
            self.train_labels: batch_labels
        }
        _, loss_val, summary_str = self.sess.run([self.train_op,self.loss,self.merged_summary_op], feed_dict=feed_dict)

        # train loss
        self.train_loss_records.append(loss_val)
        self.train_loss_k10 = sum(self.train_loss_records)/self.train_loss_records.__len__()
        if self.train_sents_num % 1000 == 0 :
            self.summary_writer.add_summary(summary_str,self.train_sents_num)
            print("{a} sentences dealed, loss: {b}"
                  .format(a=self.train_sents_num,b=self.train_loss_k10))

        # train times
        self.train_words_num += batch_inputs.__len__()
        self.train_sents_num += input_sentence.__len__()
        self.train_times_num += 1

    def cal_similarity(self,test_word_id_list,top_k=10):
        sim_matrix = self.sess.run(self.similarity, feed_dict={self.test_word_id:test_word_id_list})
        sim_mean = np.mean(sim_matrix)
        sim_var = np.mean(np.square(sim_matrix-sim_mean))
        test_words = []
        near_words = []
        for i in range(test_word_id_list.__len__()):
            test_words.append(self.vocab_list[test_word_id_list[i]])
            nearst_id = (-sim_matrix[i,:]).argsort()[1:top_k+1]
            nearst_word = [self.vocab_list[x] for x in nearst_id]
            near_words.append(nearst_word)
        return test_words,near_words,sim_mean,sim_var

    def save_model(self, save_path):

        if os.path.isfile(save_path):
            raise RuntimeError('the save path should be a dir')
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # 记录模型各参数
        model = {}
        var_names = ['vocab_size',      # int       model parameters
                     'vocab_list',      # list
                     'learning_rate',   # int
                     'word2id',         # dict
                     'embedding_size',  # int
                     'logdir',          # str
                     'win_len',         # int
                     'num_sampled',     # int
                     'train_words_num', # int       train info
                     'train_sents_num', # int
                     'train_times_num', # int
                     'train_loss_records',  # int   train loss
                     'train_loss_k10',  # int
                     ]
        for var in var_names:
            model[var] = eval('self.'+var)

        param_path = os.path.join(save_path,'params.pkl')
        if os.path.exists(param_path):
            os.remove(param_path)
        with open(param_path,'wb') as f:
            pkl.dump(model,f)

        # 记录tf模型
        tf_path = os.path.join(save_path,'tf_vars')
        if os.path.exists(tf_path):
            os.remove(tf_path)
        self.saver.save(self.sess,tf_path)

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise RuntimeError('file not exists')
        param_path = os.path.join(model_path,'params.pkl')
        with open(param_path,'rb') as f:
            model = pkl.load(f)
            self.vocab_list = model['vocab_list']
            self.vocab_size = model['vocab_size']
            self.logdir = model['logdir']
            self.word2id = model['word2id']
            self.embedding_size = model['embedding_size']
            self.learning_rate = model['learning_rate']
            self.win_len = model['win_len']
            self.num_sampled = model['num_sampled']
            self.train_words_num = model['train_words_num']
            self.train_sents_num = model['train_sents_num']
            self.train_times_num = model['train_times_num']
            self.train_loss_records = model['train_loss_records']
            self.train_loss_k10 = model['train_loss_k10']

def gen_dict(dict_size=20000):
    content = None
    with open('../word_count.pkl','rb') as f:
        content = pkl.load(f)

    cont_list = [content[x] for x in content]
    cont_list.sort(key=lambda x:x['freq'],reverse=True)
    return [x['word'] for x in cont_list[:dict_size]]

def predeal(sentence):

    # 去掉 a 块部分
    m1 = re.compile('<a.*?/a>')
    res = re.findall(m1,sentence)
    if res.__len__()>0:
        for item in res:
            m2 = re.compile('#(.*?)#')
            v = re.findall(m2,item)
            if v.__len__()==1:
                sentence = sentence.replace(item,v[0])

    # 去掉<br/>
    while '<br/>' in sentence:
        sentence = sentence.replace('<br/>','')

    return sentence

if __name__=='__main__':
    dict_size = 50000
    if os.path.exists('word_info_list.pkl'):
        with open('word_info_list.pkl','rb') as f:
            word_list = [x['word'] for x in pkl.load(f)]
    else:
        word_list = gen_dict(dict_size=dict_size)
    word_dict = {}
    for i in range(word_list.__len__()):
        word_dict[word_list[i]] = i

    # NEG版w2v 模型生成
    if os.path.exists('./model'):
        m = NEGModel(model_path='./model')
    else:
        m = NEGModel(vocab_list=word_list,embedding_size=200)

    # 连接 mongodb
    client = MongoClient('localhost',27017)
    db = client.microblog_spider
    table = db['latest_history']

    fetch_batch = 10000 # 一批从数据库读取10000条微博
    fetch_times = 0     # 统计已经读取几批
    fetch_total = 500000 # 总共要读取多少条微博
    fetch_total_times = fetch_total//fetch_batch    # 要读取的批数
    print(fetch_total_times)
    sentence_count = 0  # 已经处理的句子数目统计

    test_word_id_list = [10,20,40,80,160,320,640,7,14,28,56,112,224]
    test_word_list = [word_list[x] for x in test_word_id_list]
    print('the test words are: '+str(test_word_list) )

    batch_list = []
    batch_size = 100    # 一批交给w2v模型处理的句子数目
    while fetch_times<fetch_total_times:
        skip = (fetch_times * fetch_batch) % 1000000
        v = table.find().skip(skip).limit(fetch_batch)
        fetch_times += 1
        for x in v:
            content_list = x['dealed_text']['left_content']
            for subs in content_list:
                subs_dealed = predeal(subs)
                if subs_dealed.__len__()>0:
                    cut_res = [x for x in jieba.cut(subs_dealed,cut_all=False)]
                    while '' in cut_res:
                        cut_res.remove('')
                    valid_res = [x if x in word_dict else '' for x in cut_res]
                    while '' in valid_res:
                        valid_res.remove('')
                    # id_res = [word_dict[x] for x in valid_res]
                    batch_list.append(valid_res)
                    sentence_count += 1
                    if sentence_count % batch_size == 0:
                        m.train_by_sentence(batch_list)
                        batch_list = []
                    if sentence_count % 10000 == 0:
                        testword,nearword,sim_mean,sim_var = m.cal_similarity(test_word_id_list)
                        for i in range(test_word_id_list.__len__()):
                            print('【{w}】的近似词有： {v}'.format(w=testword[i],v=str(nearword[i])))
                        print('num_steps={v}, 相似度均值:{m}, 相似度方差:{v2}'
                              .format(v=sentence_count,m=sim_mean,v2=sim_var))

    # 将 embedding信息储存
    embed = m.sess.run(m.normed_embedding)
    word_info_list = []
    word_info_dict = {}
    for i in range(word_list.__len__()):
        info = {}
        info['word'] = word_list[i]
        info['id'] = i
        info['embedding'] = embed[i,:]
        word_info_list.append(info)
        word_info_dict[word_list[i]] = info
    with open('word_info_list.pkl','wb') as f:
        pkl.dump(word_info_list,f)
    m.save_model('./model')
