
import numpy as np
import tensorflow as tf
from pprint import pprint


# 读取文本数据
data = open('/home/multiangle/download/280.txt', 'rb').read() # should be simple plain text file
data = data.decode('gbk')
data = data.replace('\r\n\r\n','\r\n')
data = data.replace('    ','')
vocab = list(set(data))
global sampling_index
sampling_index = 0
print('the length of text is %d , the size of vocab is %d '%(data.__len__(),vocab.__len__()))

def get_fixed_pairs(batch_num=1):
    assert batch_num>0
    num_steps = 40

    global  sampling_index
    if sampling_index+num_steps+1>data.__len__():
        sampling_index = 0

    content = []
    targets = []
    for i in range(batch_num):
        line = data[sampling_index:sampling_index+num_steps]
        target = data[sampling_index+1:sampling_index+num_steps+1]
        line_ids = [vocab.index(c) for c in line]
        target_ids = [vocab.index(c) for c in target]
        content.append(line_ids)
        targets.append(target_ids)
        sampling_index += num_steps
    return content,targets

class SimpleRNNModel(object):
    def __init__(self,vocab_size,batch_size=10,num_steps=40,hidden_size=200,trainable=True):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.hidden_size = hidden_size
        self.trainable = trainable

        # 开始构建graph
        self.input_ids = tf.placeholder(tf.int32,[batch_size,num_steps],name='input_ids')
        self.target_ids =  tf.placeholder(tf.int32,[batch_size,num_steps],name='target_ids')

        # 对词典中各单词的embedding进行随机化
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding',[vocab_size,hidden_size],dtype=tf.float32)
            # input_embedding shape:[batch, num step, embedding size]
            input_embedding = tf.nn.embedding_lookup(embedding,self.input_ids)

        # 构建rnn单元，并外包一层 dropout 防止过拟合
        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size,hidden_size)
        rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell,output_keep_prob=0.8)

        # rnn初始状态为0
        self.init_state = rnn_cell.zero_state(batch_size,tf.float32)
        state = self.init_state
        outputs_embedding = []

        # rnn 接受输入，产生输出
        with tf.variable_scope('RNN'):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                state,state = rnn_cell(input_embedding[:,time_step,:],state)
                outputs_embedding.append(state)

        output = tf.reshape(tf.concat(1,outputs_embedding),[-1,hidden_size]) # [batch*num_steps, hidden_size]
        # 将embedding 形式的输出转化成logits形式
        # W = tf.Variable(tf.truncated_normal(shape=[hidden_size,vocab_size]))
        # b = tf.Variable(tf.constant(0,dtype=tf.float32,shape=[vocab_size]))
        W = tf.get_variable('W',shape=[hidden_size,vocab_size])
        b = tf.get_variable('b',shape=[vocab_size])
        output_logits = tf.matmul(output,W)+b # shape as [batch*num_steps, vocab_size]

        # 计算误差，产生梯度
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [output_logits],
            [tf.reshape(self.target_ids,[-1])],
            [tf.ones([batch_size*num_steps],dtype=tf.float32)])
        self.cost = tf.reduce_sum(loss) / batch_size
        tf.scalar_summary('perplexity',self.cost)
        self.merged_summary_op = tf.merge_all_summaries()

        if not trainable:
            return
        learning_rate = 1e-1
        tvars = tf.trainable_variables()
        grad = tf.gradients(self.cost,tvars)

        # 优化问题
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # self.train_op = optimizer.apply_gradients(zip(grad,tvars))
        self.train_op = optimizer.minimize(self.cost)

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())
        self.summary_writer = tf.train.SummaryWriter('/tmp/simple_rnn', self.sess.graph)

    def run(self):
        iter_times = 0
        while True:
            input,target = get_fixed_pairs(self.batch_size)
            cost,summary_str = self.sess.run([self.cost,self.merged_summary_op],feed_dict={self.input_ids:input,self.target_ids:target})
            iter_times += 1
            if iter_times%100==0:
                print('iter times : {t} ; perplexity: {p}'.format(t=iter_times,p=cost))
                self.summary_writer.add_summary(summary_str,iter_times)

s = SimpleRNNModel(vocab_size=vocab.__len__())
s.run()


