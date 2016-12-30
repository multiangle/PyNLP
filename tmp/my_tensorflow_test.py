from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
# import urllib
# from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from matplotlib import pyplot as plt
import pickle as pkl

# Step 1: Download the data.
url = 'http://mattmahoney.net/dc/'


# def maybe_download(filename, expected_bytes):
#     """Download a file if not present, and make sure it's the right size."""
#     if not os.path.exists(filename):
#         filename, _ = urllib.request.urlretrieve(url + filename, filename)
#     statinfo = os.stat(filename)
#     if statinfo.st_size == expected_bytes:
#         print('Found and verified', filename)
#     else:
#         print(statinfo.st_size)
#         raise Exception(
#             'Failed to verify ' + filename + '. Can you get to it with a browser?')
#     return filename

# filename = maybe_download('text8.zip', 31344016)

filename = 'text8.zip'
# Read the data into a list of strings.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

words = read_data(filename)
print('Data size', len(words))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 50000


def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
del words  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
print(dictionary)
data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ] # span 应该表示buffer的长度，即 2*skip_window + 1
    buffer = collections.deque(maxlen=span)
    for _ in range(span):  # 将长为span的单词压入buffer中
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips): # num_skips 应该是窗口的宽度
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid: # 如果选中的单词是target，则再随机换一个
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target) # 这行没看太懂
            batch[i * num_skips + j] = buffer[skip_window] # 在一个窗口长num_skip， 其batch内容不同，但是labels相同
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index]) # buffer向后延伸一个单词
        data_index = (data_index + 1) % len(data)
    return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1) # 样本数为 batch_size//num_skips, 每个样本窗口长num_skips.
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]],
          '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
