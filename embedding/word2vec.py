
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
                 vocab_size=30000,
                 embedding_size=200,
                 win_len=3, # 单边窗口长
                 num_sampled=1000,
                 model_path= None
                 ):

        # 获得模型的基本参数
        tf_vars = None
        self.batch_size     = None # 一批中数据个数, 目前是根据情况来的

        if model_path!=None:
            tf_vars = self.load_model(model_path)
        else:
            # model parameters
            self.vocab_size     = vocab_size
            self.embedding_size = embedding_size
            self.win_len        = win_len
            self.num_sampled    = num_sampled

            # train times
            self.train_words_num = 0 # 训练的单词对数
            self.train_sents_num = 0 # 训练的句子数
            self.train_times_num = 0 # 训练的次数（一次可以有多个句子）

            # train loss records
            self.train_loss_records = collections.deque(maxlen=10) # 保存最近10次的误差
            self.train_loss_k10 = 0

        self.build_graph(tf_vars)
        self.init_op()

    def init_op(self):
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)
        self.summary_writer = tf.train.SummaryWriter('/tmp/simple_rnn', self.sess.graph)

    def build_graph(self,tf_vars):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            if tf_vars==None:  # 如果是一个新模型
                self.embedding_dict = tf.Variable(
                    tf.random_uniform([self.vocab_size,self.embedding_size],-1.0,1.0)
                )
                self.nce_weight = tf.Variable(tf.truncated_normal([self.vocab_size, self.embedding_size],
                                                             stddev=1.0/math.sqrt(self.embedding_size)))
                self.nce_biases = tf.Variable(tf.zeros([self.vocab_size]))
            else:   # 如果是从已有模型中载入：
                self.embedding_dict = tf.Variable(tf_vars[0])
                self.nce_weight = tf.Variable(tf_vars[1])
                self.nce_biases = tf.Variable(tf_vars[2])

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
            tf.scalar_summary('loss',self.loss)

            # 根据 nce loss 来更新梯度和embedding
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(self.loss)  # 训练操作

            # 计算与指定若干单词的相似度
            self.test_word_id = tf.placeholder(tf.int32,shape=[None])
            vec_l2_model = tf.sqrt(  # 求各词向量的L2模
                tf.reduce_sum(tf.square(self.embedding_dict),1,keep_dims=True)
            )

            avg_l2_model = tf.reduce_mean(vec_l2_model)
            tf.scalar_summary('avg_vec_model',avg_l2_model)

            norm_vec = self.embedding_dict / vec_l2_model
            test_embed = tf.nn.embedding_lookup(norm_vec, self.test_word_id)
            self.similarity = tf.matmul(test_embed, norm_vec, transpose_b=True)

            # 变量初始化
            self.init = tf.global_variables_initializer()

            self.merged_summary_op = tf.merge_all_summaries()

    def train_by_sentence(self, input_sentence=[]):  #  input_sentence: [sub_sent1, sub_sent2, ...]
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
                        batch_inputs.append(sent[index])
                        batch_labels.append(sent[i])
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
            print("{a} sentences dealed, loss: {b}".format(a=self.train_sents_num,b=self.train_loss_k10))

        # train times
        self.train_words_num += batch_inputs.__len__()
        self.train_sents_num += input_sentence.__len__()
        self.train_times_num += 1

    def cal_similarity(self,test_word_id_list):
        sim_matrix = self.sess.run(self.similarity, feed_dict={self.test_word_id:test_word_id_list})
        return sim_matrix

    def save_model(self, save_path):
        # 记录模型各参数
        model = {}
        var_names = ['vocab_size',      # int       model parameters
                     'embedding_size',  # int
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
        ops = [self.embedding_dict,
               self.nce_weight,
               self.nce_biases
               ]
        res = self.sess.run(ops)
        model['embedding_dict'] = res[0]
        model['nce_weight'] = res[1]
        model['nce_biases'] = res[2]

        # 写入
        if os.path.exists(save_path):
            os.remove(save_path)
        with open(save_path,'wb') as f:
            pkl.dump(model,f)

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise RuntimeError('file not exists')
        with open(model_path,'rb') as f:
            model = pkl.load(f)
            self.vocab_size = model['vocab_size']
            self.embedding_size = model['embedding_size']
            self.win_len = model['win_len']
            self.num_sampled = model['num_sampled']
            self.train_words_num = model['train_words_num']
            self.train_sents_num = model['train_sents_num']
            self.train_times_num = model['train_times_num']
            self.train_loss_records = model['train_loss_records']
            self.train_loss_k10 = model['train_loss_k10']
            ret = []
            ret.append(model['embedding_dict'])
            ret.append(model['nce_weight'])
            ret.append(model['nce_biases'])
            return ret

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

dict_size = 50000
# 生成词典
if os.path.exists('word_info_list.pkl'):
    f = open('word_info_list.pkl','rb')
    v = pkl.load(f)
    word_list = [x['word'] for x in v]
else:
    word_list = gen_dict(dict_size=dict_size)
word_dict = {}
for i in range(word_list.__len__()):
    word_dict[word_list[i]] = i

# NEG版w2v 模型生成
if os.path.exists('model'):
    m = NEGModel(model_path='model')
else:
    m = NEGModel(vocab_size=dict_size)

# 连接 mongodb
client = MongoClient('localhost',27017)
db = client.microblog_spider
table = db['latest_history']

fetch_batch = 10000 # 一批从数据库读取10000条微博
fetch_times = 0     # 统计已经读取几批
fetch_total = 1000000 # 总共要读取多少条微博
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
                id_res = [word_dict[x] for x in valid_res]
                batch_list.append(id_res)
                sentence_count += 1
                if sentence_count % batch_size == 0:
                    m.train_by_sentence(batch_list)
                    batch_list = []
                if sentence_count % 10000 == 0:
                    sim = m.cal_similarity(test_word_id_list)
                    top_k = 10
                    for i in range(test_word_id_list.__len__()):
                        nearst_id = (-sim[i,:]).argsort()[1:top_k+1]
                        nearst_word = [word_list[x] for x in nearst_id]
                        print('【{w}】的近似词有： {v}'.format(w=word_list[test_word_id_list[i]],v=str(nearst_word)))

# 将 embedding信息储存
embed = m.sess.run(m.embedding_dict)
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
with open('word_info_dict.pkl','wb') as f:
    pkl.dump(word_info_dict,f)
m.save_model('model')
