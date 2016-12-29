import numpy as np
import tensorflow as tf

a = tf.Variable(np.array([[1,2,3,4,5],[6,7,8,9,10]]),dtype=tf.float32)
b = tf.Variable(np.array([-2,-1,0.5,1,2]),dtype=tf.float32)
l2 = tf.sqrt(tf.reduce_sum(tf.square(a),axis=1,keep_dims=True))
# l2m = tf.expand_dims(l2,1)
l2x = tf.matmul(l2,tf.ones([1,5]))
# v = tf.divide(a,l2x)
v = tf.divide(a,l2x)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

print(sess.run(a))
print(sess.run(l2))
print(sess.run(a/l2))
new_l2 = tf.sqrt(tf.reduce_sum(tf.square(v),axis=1,keep_dims=True))
print(sess.run(new_l2))
