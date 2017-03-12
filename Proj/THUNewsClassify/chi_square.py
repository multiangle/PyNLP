
import tensorflow as tf
import numpy as np
import pickle as pkl
from pprint import pprint

class ChiSquareCalculator():
    def __init__(self,item_size,label_size):
        # item size: 词语的个数
        # label size: 词语所属类别个数
        self.item_size = item_size
        self.label_size = label_size
        self.count_matrix = np.zeros([item_size,label_size],dtype=np.int32)

    def add(self,item_id,label_id,count=1):
        assert item_id<self.item_size
        assert label_id<self.label_size
        self.count_matrix[item_id][label_id] += count

    def cal(self):
        label_count = np.sum(self.count_matrix,axis=0,keepdims=True)  # the sum of items in each label

        self.item_num = np.sum(label_count)   # the sum of num of items
        self.label_pd = label_count / self.item_num # label possibility distribution

        line_count = np.sum(self.count_matrix,axis=1,keepdims=True) # the sum of items in each line
        line_pd = self.count_matrix / (line_count+1)  # item_size * label_size

        self.chi_value = np.sum(np.square(line_pd-self.label_pd)/(self.label_pd),
                                 axis=1)

if __name__=='__main__':
    def pick_valid_word(word_info_list, dict_size):
        word_info_list.sort(key=lambda x:x['count'],reverse=True)
        word_info_list = word_info_list[:dict_size]
        word2id = {}
        id2word = {}
        for line in word_info_list:
            word = line['word']
            id = line['id']
            word2id[word] = id
            id2word[id] = word
        return word2id,id2word

    with open('file_info_list.pkl','rb') as f:
        file_info_list = pkl.load(f)
    with open('word_list_path.pkl','rb') as f:
        word_info_list = pkl.load(f)
        word2id,id2word = pick_valid_word(word_info_list,100)
        print(word2id)
        print(id2word)
        print(len(word2id))
    c = ChiSquareCalculator(item_size=20,label_size=10)













        