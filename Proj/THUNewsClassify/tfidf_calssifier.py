from Proj.THUNewsClassify.simple_add_classifier import SimpleClassifier
from Proj.THUNewsClassify.util import read_text,rm_words,gen_balance_samples,pick_valid_word,pick_valid_word_chisquare,gen_balance_samples_withid
from TextDeal import isStopWord,isValidWord
import jieba
import numpy as np
import tensorflow as tf
import math,random,collections
from pprint import pprint
import pickle as pkl
from matplotlib import pyplot as plt

# def gen_test_sample(file_info_list)

if __name__=='__main__':
    dict_size = 50000

    # 根据卡方和词频选出若干个词，产生 word <-> id映射，以及id->id映射
    with open('word_list_path_with_docfreq.pkl','rb') as f:
        word_info_list = pkl.load(f)
        full_word2id = {}
        for info in word_info_list:
            full_word2id[info['word']] = info['id']
        word2id,id2word = pick_valid_word_chisquare(word_info_list,dict_size=30000,s1_size=dict_size)
        id_old2new = {}
        for word in word2id:
            new_id = word2id[word]
            old_id = full_word2id[word]
            id_old2new[old_id] = new_id

    with open('file_info_list.pkl','rb') as f:
        file_info_list = pkl.load(f)
        file_full_nums = len(file_info_list)
    label_list = ['娱乐', '股票', '体育', '科技', '房产', '社会', '游戏', '财经', '时政', '家居', '彩票', '教育', '时尚', '星座']

    # 产生tf - idf 权重
    weights = np.ones([dict_size])
    for word in word2id:
        word_info = word_info_list[full_word2id[word]]
        part_tf = 1.0/word_info['count']
        if 'doc_freq' in word_info:
            part_idf =  math.log(file_full_nums/word_info['doc_freq'])
        else:
            part_idf = math.log(file_full_nums/word_info['count'])
        weights[word2id[word]] = part_tf * part_idf

    # 开始训练
    with open('THUCNews_fullid_type.pkl','rb') as f:
        contents_idtype = pkl.load(f)

    file_infos = gen_balance_samples_withid(file_info_list,label_list,balance_index=3)
    m = SimpleClassifier(label_size=len(label_list),embed_size=dict_size)
    num_steps = 200000
    train_nums = math.ceil(len(file_infos)*0.9)
    loss_deque = collections.deque(maxlen=2000)
    err_deque = collections.deque(maxlen=2000)
    sub_deque = [collections.deque(maxlen=500) for _ in range(len(label_list))]
    for i in range(num_steps):
        file_info = file_infos[i%train_nums]
        context_id = file_info['id']
        file_path = file_info['path']
        label = file_info['label']
        if context_id>=len(contents_idtype):
            continue
        content = contents_idtype[context_id]['content']
        content = sum(content,[])
        valid_content = filter(lambda x:x in id_old2new,content)
        valid_content = [id_old2new[x] for x in valid_content]
        count_vector = np.zeros([dict_size])
        for id in valid_content:
            count_vector[id] += 1
        tfidf_vector = np.multiply(count_vector,weights)
        feed_dict = {
            m.train_input:[tfidf_vector],
            m.train_label:[label_list.index(label)]
        }
        _,err_num,loss = m.sess.run([m.train_op,m.error_num,m.loss],feed_dict=feed_dict)
        loss_deque.append(loss)
        err_deque.append(err_num)
        sub_deque[label_list.index(label)].append(err_num)
        if i%500==0:
            avg_loss = np.mean(loss_deque)
            avg_accu = 1-np.mean(err_deque)
            avg_sub = [1-np.mean(x) for x in sub_deque]
            print(i,'\t',avg_loss,'\t',avg_accu)
            print(str(avg_sub))

