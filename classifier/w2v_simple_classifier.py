
import tensorflow as tf
import numpy as np
import pickle as pkl
import os.path as path
import os
import math
from pprint import pprint
import collections
from pymongo import MongoClient
import jieba
import TextDeal

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
            logit_sigmoid = tf.sigmoid(logits)
            self.predict = tf.to_int32(tf.argmax(logit_sigmoid,axis=1))

            self.error_times = tf.count_nonzero(tf.subtract(self.predict,self.batch_label))
            self.accuracy = (self.batch_size - self.error_times) / self.batch_size


            self.loss = tf.nn.seq2seq.sequence_loss([logit_sigmoid],
                                               [self.batch_label],
                                               [tf.ones([batch_size])])

            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=5.0).minimize(self.loss)

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
                word_embed_list = []
                for word in line:
                    id = self.word2id.get(word)
                    if id:
                        word_embed_list.append(self.embedding[id,:])
                sentence_embed = np.mean(word_embed_list,axis=0)
                batch_sentence_embed.append(sentence_embed)
            batch_input = np.array(batch_sentence_embed)
            batch_label = np.array(batch_label)
            # print(batch_label)
            feed_dict = {self.batch_embed_input:batch_input, self.batch_label:batch_label}
            _, loss, accuracy, summary_str = self.sess.run([self.train_op,
                                                            self.loss,
                                                            self.accuracy,
                                                            self.merge_summary_op],
                                                           feed_dict=feed_dict)
            self.train_loss_records.append(loss)
            self.train_accu_records.append(accuracy)
            if self.train_sent_num%10 == 0:
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


# 读取词库信息
f = open('../embedding/word_info_list.pkl','rb')
word_info_list = pkl.load(f)
f.close()
word_list = [x['word'] for x in word_info_list]
word_set = set(word_list)
word_dict = {}
for word_info in word_info_list:
    word_dict[word_info['word']] = word_info['id']
embedding = [x['embedding'] for x in word_info_list]
embedding_dict = np.array(embedding)


# 读取数据
client = MongoClient('localhost',27017)
db = client['microblog_classify']
table = db.test
row_data = [x for x in table.find()]

# 对数据进行拼接
concat_data = []
for line in row_data:
    emotion = line.get('emotion',None)
    if not emotion:
        raise RuntimeError('invalid emotion value')
    tmp_content = []
    tmp_content += line.get('left_content',[])
    tmp_content += line.get('retweeted_left_content',[])
    tmp_sent = ''
    for sub_sen in tmp_content:
        tmp_sent += sub_sen
    while '' in tmp_content:
        tmp_content.remove('')

    category = line.get('category',[])
    if type(category)!=list:
        category = [category]
    for cat in category:
        tmp_pack = {}
        tmp_pack['text'] = tmp_sent
        tmp_pack['category'] = cat
        tmp_pack['emotion'] = emotion
        concat_data.append(tmp_pack)

# 对文本进行预处理
def predeal(sentence):
    if sentence.__len__()<3:
        return []
    sentence = TextDeal.removeLinkOnly(sentence)
    words = [x for x in jieba.cut(sentence, cut_all=False)]
    valid_words = []
    my_own_stop = ['【','】','《','》','@']
    for word in words:
        if TextDeal.isStopWord(word):
            continue
        if word in my_own_stop:
            continue
        if word not in word_set:
            continue
        valid_words.append(word)
    return  valid_words

def cal_l2_model(np_vector):
    return np.sqrt(np.sum(np.square(np_vector)))

dealed_data = []
emotion_list = []
category_list = []
for item in concat_data:
    content = item['text']
    if content.__len__()<3:
        continue
    valid_words = predeal(content)
    if valid_words.__len__()==0:
        continue
    item['text'] = valid_words
    dealed_data.append(item)

    if item['emotion'] not in emotion_list:
        emotion_list.append(item['emotion'])
    if item['category'] not in category_list:
        category_list.append(item['category'])

# 建立模型并训练
classifier = SimpleClassifier(word_list=word_list,
                              embedding = embedding_dict,
                              tag_num = category_list.__len__())
data_categorys = [category_list.index(x['category']) for x in dealed_data]
data_sentences = [x['text'] for x in dealed_data]
input_batch = 50
sample_num = concat_data.__len__()
intpu_batch_times = sample_num // input_batch
for t in range(200):
    for i in range(intpu_batch_times):
        start_index = i*input_batch
        end_index = (i+1)*input_batch
        classifier.train_by_sentences(data_sentences[start_index:end_index],
                                      data_categorys[start_index:end_index])









