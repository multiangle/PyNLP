from Proj.THUNewsClassify.util import \
    pick_valid_word_chisquare, pick_valid_word_chisquare_concat, \
    gen_balance_samples_withid
import pickle as pkl
import numpy as np
import math, sys, os, gc

def gen_data():
    dict_size = 50000
    chi_size = 10000
    # 根据卡方和词频选出若干个词，产生 word <-> id映射，以及id->id映射
    with open('word_list_path_with_docfreq.pkl', 'rb') as f:
        full_word_info_list = pkl.load(f)
    full_word2id = {}
    for info in full_word_info_list:
        full_word2id[info['word']] = info['id']
    word2id,id2word = pick_valid_word_chisquare_concat(full_word_info_list,dict_size=chi_size,s1_size=dict_size)
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
    weights = np.ones([chi_size])
    for word in word2id:
        word_info = full_word_info_list[full_word2id[word]]
        part_tf = 1.0/word_info['count']
        if 'doc_freq' in word_info:
            part_idf =  math.log(file_full_nums/word_info['doc_freq'])
        else:
            part_idf = math.log(file_full_nums/word_info['count'])
        weights[word2id[word]] = part_tf * part_idf

    vector_list = []
    out_list = []
    file_path = 'THUCNews_balance_id_type.pkl'
    reuse = 1
    if os.path.exists(file_path) and reuse==1:
        with open(file_path, 'rb') as f:
            tmp = pkl.load(f)
            vector_list = tmp[0]
            out_list = tmp[1]
    else:
        print(file_path, 'not exists!')
        # 载入数据
        with open('THUCNews_fullid_type.pkl','rb') as f:
            contents_idtype = pkl.load(f)
        file_infos = gen_balance_samples_withid(file_info_list,label_list,balance_index=3)
        print(len(file_infos))

        for i, file_info in enumerate(file_infos):
            context_id = file_info['id']
            label = file_info['label']
            if context_id>=len(contents_idtype):
                continue
            content = contents_idtype[context_id]['content'] # 是[[],[],[]]形式
            content = sum(content,[])  # 拼接起来
            valid_content = filter(lambda x:x in id_old2new,content)
            valid_content = [id_old2new[x] for x in valid_content]
            count_vector = np.zeros([chi_size])
            for id in valid_content:
                count_vector[id] += 1
            tfidf_vector = np.multiply(count_vector,weights)
            # print(sys.getsizeof(tfidf_vector))
            vector_list.append(tfidf_vector)
            # print(i, sys.getsizeof(vector_list))
            out_list.append(label_list.index(label))
        with open(file_path, 'wb') as f:
            tmp = [vector_list, out_list]
            pkl.dump(tmp, f)
    input_data = np.array(vector_list)
    print(out_list)

def train(input, output):
    pass

if __name__=='__main__':
    input, output = gen_data()

    train(input, output)

