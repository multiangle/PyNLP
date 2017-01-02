import tensorflow as tf
import numpy as np
from embedding.word2vec import NEGModel
import os
from pprint import pprint
import random
import jieba
import TextDeal
import pickle as pkl

def rm_chars(sent):
    new_sent = ''
    warn_chars = [
        '\u3000',
        '\n',
        '\xa0',
        '□',
        '■',
        '●',
    ]
    for char in sent:
        if char in warn_chars:
            continue
        new_sent += char
    return new_sent

def read_text(file_path):
    lines = []
    with open(file_path) as f:
        line = f.readline()
        while line:
            dealed_line = rm_chars(line)
            if dealed_line.__len__()>0:
                lines.append(dealed_line)
            line = f.readline()
        return lines

def rm_words(word_list):
    new_words = []
    for word in word_list:
        if not TextDeal.isValidWord(word):
            continue
        banned_word = [' ','，',',','.']
        if word in banned_word:
            continue
        new_words.append(word)
    return new_words

if __name__=='__main__':
    data_root_path = '/media/multiangle/F/DataSet/THUCNews/THUCNewsPart'
    files_info_list_path = 'file_info_list.pkl'  # step1
    word_list_path = 'word_list_path.pkl'
    word_dict_path = 'word_dict_path.pkl'
    neg_model_path = './neg_model'
    label_list = os.listdir(data_root_path)
    # step 1 生成所有文件列表
    if os.path.exists(files_info_list_path):
        with open(files_info_list_path,'rb') as f:
            file_info_list = pkl.load(f)
    else:
        cate_list = os.listdir(data_root_path)
        file_info_list = []
        for cate in cate_list:
            tmp_path = os.path.join(data_root_path,cate)
            files = os.listdir(tmp_path)
            for file_path in files:
                abs_path = os.path.join(tmp_path,file_path)
                tmp_info = {}
                tmp_info['path'] = abs_path
                tmp_info['label'] = cate
                # tmp_info['size'] = os.path.getsize(abs_path)
                file_info_list.append(tmp_info)
        random.shuffle(file_info_list)
        with open(files_info_list_path,'wb') as f:
            pkl.dump(file_info_list,f)

    # step 2 统计单词，生成词典
    print(os.path.exists(files_info_list_path))
    print(os.path.abspath('.'))
    print(os.path.exists(word_list_path))
    print(os.path.exists('word_list_path.pkl'))
    print(os.path.exists('./word_list_path.pkl'))
    if os.path.exists(word_list_path) and os.path.exists(word_dict_path):
        with open(word_list_path,'rb') as f:
            word_list = pkl.load(f)
        with open(word_dict_path,'rb') as f:
            word_dict = pkl.load(f)
    else:
        word_list = []
        word_dict = {}
        count = 0
        for file_info in file_info_list:
            count += 1
            print(count)
            tmp_path = file_info['path']
            lines = read_text(tmp_path)
            context = "".join(lines)
            words = [x for x in jieba.cut(context,cut_all=False)]
            valid_words = rm_words(words)
            for word in valid_words:
                if word not in word_dict:
                    tmp_v = {}
                    tmp_v['word'] = word
                    tmp_v['id'] = word_list.__len__()
                    tmp_v['count'] = 0
                    tmp_v['sub_count'] = {}
                    word_dict[word] = tmp_v
                    word_list.append(tmp_v)
                else:
                    word_dict[word]['count'] += 1
                lable = file_info['label']
                if lable in word_dict[word]['sub_count']:
                    word_dict[word]['sub_count'][lable] += 1
                else:
                    word_dict[word]['sub_count'][lable] = 1
                    # print(word_dict[word])

        with open(word_list_path,'wb') as f:
            pkl.dump(word_list,f)
        with open(word_dict_path,'wb') as f:
            pkl.dump(word_dict,f)

    # step 3 建立neg模型
    word_list.sort(key=lambda x:x['count'],reverse=True)
    valid_word_info_list = word_list[:50000]
    valid_word_list = [x['word'] for x in valid_word_info_list]
    if os.path.exists(neg_model_path):
        neg = NEGModel(model_path=neg_model_path)
    else:
        neg = NEGModel(vocab_list=valid_word_list,
                       learning_rate=0.001,
                       num_sampled=100,
                       win_len=3,logdir='/tmp/THU_w2v')

    count = 0
    for file_info in file_info_list:
        count += 1
        file_path = file_info['path']
        file_label = file_info['label']
        lines = read_text(file_path)
        context = "".join(lines)
        sents = context.split('。')
        words_in_sents = []
        for sent in sents:
            words = [x for x in jieba.cut(context,cut_all=False)]
            valid_words = rm_words(words)
            words_in_sents.append(valid_words)
        batch_size = 1
        times = words_in_sents.__len__() //batch_size
        for i in range(times):
            start = i*batch_size
            end = min((i+1)*batch_size,words_in_sents.__len__())
            neg.train_by_sentence(words_in_sents[start:end])
        # neg.train_by_sentence([rm_words(jieba.cut(context,cut_all=False))])
        if count%250==0:
            test_word_id = [2,4,8,16,32,64,128,150,200,250,300]
            test_word,near_word = neg.cal_similarity(test_word_id_list=test_word_id,top_k=20)
            for i in range(test_word.__len__()):
                print('【{w}】的近似词有： {v}'.format(w=test_word[i],v=str(near_word[i])))

        if count%250 ==0:
            print('count={s}, saving model=='.format(s=count))
            neg.save_model(neg_model_path)
            norm_embedding = neg.sess.run(neg.normed_embedding)
            with open('THUCNews.pkl','wb') as f:
                pkl.dump(norm_embedding,f)




