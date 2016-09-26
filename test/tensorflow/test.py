import tensorflow as tf


tensor = tf.zeros(shape=[1,2])
variable = tf.Variable(tensor)
sess = tf.InteractiveSession()
# print(sess.run(variable))  # 会报错
sess.run(tf.initialize_all_variables())
print(sess.run(variable))