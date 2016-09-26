import numpy as np
import random
import math
import tensorflow as tf
import matplotlib.pyplot as plt


def gen_sample():
    data = []
    radius = [0,50]
    for i in range(1000):  # 生成10k个点
        catg = random.randint(0,1)  # 决定分类
        r = random.random()*10
        arg = random.random()*360
        len = r + radius[catg]
        x_c = math.cos(math.radians(arg))*len
        y_c = math.sin(math.radians(arg))*len
        x = random.random()*30 + x_c
        y = random.random()*30 + y_c
        data.append([x,y,catg])
    return data

def plot_dots(data):
    data_asclass = [[] for i in range(2)]
    for d in data:
        data_asclass[int(d[2])].append((d[0],d[1]))
    colors = ['r.','b.','r.','b.']
    for i,d in enumerate(data_asclass):
        # print(d)
        nd = np.array(d)
        plt.plot(nd[:,0],nd[:,1],colors[i])
    plt.draw()

if __name__=='__main__':
    data = gen_sample()
    data = np.array(data)
    input_dim = 2
    output_dim = 2
    hidden_size = 100

    x = tf.placeholder(tf.float32,[None, input_dim],name='input')
    y = tf.placeholder(tf.float32,[None, output_dim],name='output')

    # with tf.name_scope('hidden') as scope:
    W_h = tf.Variable(tf.truncated_normal([input_dim,hidden_size],stddev=0.1),name='weights')
    b_h = tf.Variable(tf.constant(0.1, shape=[hidden_size],name='bias'))
    h = tf.Variable(tf.zeros(shape=[input_dim,hidden_size],name='h'))
    # with tf.name_scope('generate') as scope:
    W_y = tf.Variable(tf.truncated_normal([hidden_size,output_dim],stddev=0.1),name='weights')
    b_y = tf.Variable(tf.constant(0.1, shape=[output_dim],name='bias'))
    # print(W_y.name)


    h = tf.nn.sigmoid((tf.matmul(x,W_h)+b_h))  # mul 是点乘， matmul才是矩阵乘法
    a = tf.nn.softmax(tf.matmul(h,W_y)+b_y)

    loss = tf.reduce_mean(tf.square(a-y))
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()
    sess.run(init)

    # 训练可视化
    tf.scalar_summary("loss", loss)
    merged_summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter('/tmp/scd_logs', sess.graph)

    tags = data[:,2]
    output = np.zeros([data.__len__(),2])
    for i,tag in enumerate(tags):
        output[i,tag] = 1
    train_data = data[0:800,0:2]
    train_out = output[0:800,:]
    test_data = data[800:1000,0:2]
    test_out = output[800:1000,:]

    correct_prediction = tf.equal(tf.argmax(a,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    err_ratio = []
    for i in range(3000):
        batch = 10
        start = random.randint(0,train_data.__len__()-batch-1)
        input = train_data[start:start+batch,:]
        output = train_out[start:start+batch,:]
        sess.run(train, feed_dict={x:input, y:output})
        if i%100==0:
            err_ratio.append(1-accuracy.eval(feed_dict={x:test_data,y:test_out}))
            summ_str = sess.run(merged_summary_op,feed_dict={x:test_data,y:test_out})
            summary_writer.add_summary(summ_str,i)



    res = sess.run(tf.argmax(a,1),feed_dict={x:data[:,0:2]})

    plt.subplot(2,2,1)
    plot_dots(data)
    plt.draw()
    for i,r in enumerate(res):
        data[i,2] = r
    plt.subplot(2,2,2)
    plot_dots(data)
    plt.subplot(2,1,2)
    plt.plot(err_ratio)
    plt.show()