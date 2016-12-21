
import tensorflow as tf
import math

# class NEGModel():
#     def __init__(self,
#                  vocab_size=30000,
#                  embedding_size=200,
#                  batch_size=1,
#                  win_len=5,
#                  num_sampled=64
#                  ):
#
#         self.train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
#         self.train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
#
#         self.embedding_dict = tf.Variable(tf.random_uniform([vocab_size,embedding_size],-1.0,1.0))
#         nce_weight = tf.Variable(tf.truncated_normal([vocab_size, embedding_size],
#                                                      stddev=1.0/math.sqrt(embedding_size)))
#         nce_biases = tf.Variable(tf.zeros([vocab_size]))
#
#         embed = tf.nn.embedding_lookup(self.embedding_dict, self.train_inputs) # batch_size
#         loss = tf.reduce_mean(
#             tf.nn.nce_loss(
#                 weights = nce_weight,
#                 biases = nce_biases,
#                 labels = self.train_labels,
#                 inputs = embed,
#                 num_sampled = num_sampled,
#                 num_classes = vocab_size
#             )
#         )
#
#         optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

    # def run(self):
