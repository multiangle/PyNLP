import tensorflow as tf

sess = tf.InteractiveSession()
a = tf.get_variable('a',shape=[2,5])
b = a
a_drop = tf.nn.dropout(a,0.8)
sess.run(tf.initialize_all_variables())
print(sess.run(b))
print(sess.run(a_drop))

