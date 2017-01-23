import tensorflow as tf
import numpy as np
import math
import pickle as pkl
import jieba
import os
import TextDeal
import collections
from Proj.THUNewsClassify.util import pick_valid_word,\
    pick_valid_word_chisquare,read_text,rm_words,rm_stop_words
from Proj.THUNewsClassify.LTM import LTMCell

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
        print('pin2')
        self.buildGraph()
        print('pin3')
        self.sess = tf.Session(graph=self.graph)
        print('pin4')
        self.sess.run(self.init_op)

    def buildGraph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputs = tf.placeholder(tf.float32,[self.batch_size,self.max_depth,self.embed_size]) # inputs: num_step * embed_size
            self.seq_len = tf.placeholder(tf.int32)
            self.label = tf.placeholder(tf.int32,[1])
            label_float = tf.cast(self.label,tf.float32)
            label_matrix = tf.diag(tf.ones(self.label_size))
            embed_label = tf.nn.embedding_lookup(label_matrix,self.label)
            print('pin2.1')
            # input_list = list(tf.split(0,self.max_depth,expand_inputs))
            input_list = tf.unpack(self.inputs,axis=1)    # [[1,embed_size,]...,[1,embed_size]]
            print('pin2.2')
            # BasicRNNCell: [num_units, input_size, ...]
            # self.rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.hidden_size,self.embed_size)
            # self.rnn_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size,self.embed_size,state_is_tuple=True)
            self.rnn_cell = LTMCell(self.hidden_size,self.embed_size,state_is_tuple=True)
            self.rnn_cell = tf.nn.rnn_cell.DropoutWrapper(self.rnn_cell,output_keep_prob=0.9)
            print('pin2.3')
            init_stat = self.rnn_cell.zero_state(1,tf.float32)
            output_embedding,states = tf.nn.rnn(self.rnn_cell,input_list,
                                                initial_state=init_stat,
                                                sequence_length=self.seq_len)

            # state = init_stat
            # states = []
            # with tf.variable_scope('RNN'):
            #     for time_step in range(max_depth):
            #         if tf.equal(time_step,self.seq_len):
            #             break
            #         if time_step>0:
            #             tf.get_variable_scope().reuse_variables()
            #         m,state = self.rnn_cell(input_list[time_step,:],state)
            #         states.append(state)
            # final_output = states[-1][0]

            print('pin2.4')
            final_output = states[-1] # final_output : [1,hidden_size]
            print(final_output.get_shape())

            weight = tf.Variable(tf.truncated_normal([self.label_size,self.hidden_size],
                                                     stddev=1.0/math.sqrt(self.hidden_size)))
            biase = tf.Variable(tf.zeros([self.label_size]))

            tmp_y = tf.matmul(final_output,weight,transpose_b=True) + biase
            tmp_g = tf.sigmoid(tmp_y)

            self.predict = tf.cast(tf.argmax(tmp_g,axis=1),tf.float32)
            self.error_num = tf.count_nonzero(label_float-self.predict)

            tiny_v = 0.0001
            self.loss =  -tf.reduce_mean(embed_label*tf.log(tmp_g+tiny_v) + (1-embed_label)*tf.log(1+tiny_v-tmp_g))

            self.train_op = tf.train.AdagradOptimizer(learning_rate=1).minimize(self.loss)
            self.init_op = tf.global_variables_initializer()

if __name__=='__main__':
    with open('word_list_path.pkl','rb') as f:
        word_info_list = pkl.load(f)
        full_word2id = {}
        for word_info in word_info_list:
            word = word_info['word']
            id = word_info['id']
            full_word2id[word] = id
        word2id,id2word = pick_valid_word_chisquare(word_info_list,10000)
        id_full2new = {}
        for word in word2id:
            new_id = word2id[word]
            old_id = full_word2id[word]
            if new_id and old_id:
                id_full2new[old_id] = new_id
    with open('THUCNews.pkl','rb') as f:
        embedding = pkl.load(f)
    with open('file_info_list.pkl','rb') as f:
        file_info_list = pkl.load(f)
    label_list = []
    for info in file_info_list:
        label = info['label']
        if label not in label_list:
            label_list.append(label)
    with open('THUCNews_fullid_type.pkl','rb') as f:
        content_fullid_type = pkl.load(f)
    print('文件读取完毕')

    print(label_list)
    label_size = label_list.__len__()
    embed_size = embedding[0].__len__()

    # 遍历一遍，筛选出有效词，以【new】 id形式保存，并记录每篇文档中单词数目
    pool_size = 2
    valid_content_list = []
    lens = []
    max_depth = -1
    for i,info in enumerate(content_fullid_type):
        content = info['content']
        content = sum(content,[])
        valid_content = list(filter(lambda x:x in id_full2new,content))
        valid_content = [id_full2new[x] for x in valid_content]
        # valid_words = [id2word[x] for x in valid_content]
        # valid_words = rm_stop_words(valid_words)
        # valid_content = [word2id[x] for x in valid_words]
        pooled_len = math.ceil(len(valid_content)/ pool_size)
        if max_depth<pooled_len:
            max_depth = pooled_len
        valid_content_list.append(valid_content)
        lens.append(len(valid_content))
    print('有效序列的最大长度为: ',max_depth)
    print('有效序列的平均长度为: ',sum(lens)/len(lens))

    # # 对每个文本生成长为max_depth的embed序列，多余部分补零,丢入rnn模型
    print('pin0')
    c = RNNClassifier(label_size=len(label_list),max_depth=max_depth,embed_size=embed_size,hidden_size=400)
    print('pin1')
    count = 0
    loss_deque = collections.deque(maxlen=200)
    len_deque = collections.deque(maxlen=200)
    err_deque = collections.deque(maxlen=200)
    err_sub_deque = [collections.deque(maxlen=200) for _ in range(label_size)]

    for i,context in enumerate(valid_content_list):
         input = np.zeros([max_depth,embed_size],dtype=np.float32)
         label = content_fullid_type[i]['label']
         label_id = label_list.index(label)
         seq_len = len(context)
         tmp_embed_container = []
         line_id = 0
         for count,word_id in enumerate(context):
             tmp_embed_container.append(embedding[word_id,:])
             if len(tmp_embed_container)==pool_size:
                 tmp_matrix = np.array(tmp_embed_container)
                 input[line_id,:] = np.max(tmp_matrix,axis=0)
                 line_id += 1
                 tmp_embed_container = []
         if len(tmp_embed_container)>0:
             tmp_matrix = np.array(tmp_embed_container)
             input[line_id,:] = np.max(tmp_matrix,axis=0)
         feed_dict = {
             c.inputs : [input],
             c.seq_len : seq_len,
             c.label : [label_id]
         }
         _,loss,err_num = c.sess.run([c.train_op,c.loss,c.error_num],feed_dict=feed_dict)
         loss_deque.append(loss)
         len_deque.append(seq_len)
         err_deque.append(err_num)
         err_sub_deque[label_id].append(err_num)
         if i%200==0:
             print('{i}篇文章\tloss:{l}\t准确率:{a}\t平均长度:{l2}'.format(i=i,l=np.mean(loss_deque),a=1-np.mean(err_deque),l2=np.mean(len_deque)))
             print(str([1-np.mean(x) for x in err_sub_deque]))
         count += 1
