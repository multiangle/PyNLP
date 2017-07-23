from Proj.THUNewsClassify.util import \
    pick_valid_word_chisquare, pick_valid_word_chisquare_concat, \
    gen_balance_samples_withid
import pickle as pkl
import numpy as np
import math, sys, os, gc
import matplotlib.pyplot as plt
from scipy import sparse
from collections import Counter

from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

def gen_data_sparse():
    dict_size = 150000
    chi_size = 30000
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

    # 载入文件信息
    balance_index = 3
    with open('THUCNews_fullid_type.pkl','rb') as f:
        contents_idtype = pkl.load(f)
    # file_infos = gen_balance_samples_withid(file_info_list,label_list,balance_index=balance_index)
    file_infos = file_info_list
    print("file num is {}, balance index is {}".format(len(file_infos), balance_index))
    print(len(file_infos))

    # x_data = sparse.dok_matrix((len(file_infos), chi_size))
    out_list = []
    test_out_list = []
    file_path = 'THUCNews_balance_id_type.pkl'
    reuse = 0
    if os.path.exists(file_path) and reuse==1:
        with open(file_path, 'rb') as f:
            tmp = pkl.load(f)
            x_data = tmp[0]
            out_list = tmp[1]
    else:
        print(file_path, 'not exists!')
        cols = []
        rows = []
        data = []
        test_cols = []
        test_rows = []
        test_data = []
        valid_info_count = 0

        train_ratio = 0.9
        train_num = math.ceil(len(file_infos) * train_ratio)

        for i, file_info in enumerate(file_infos):
            context_id = file_info['id']
            label = file_info['label']
            if context_id>=len(contents_idtype):
                continue

            content = contents_idtype[context_id]['content'] # 是[[],[],[]]形式
            content = sum(content,[])  # 拼接起来
            valid_content = filter(lambda x:x in id_old2new,content)
            valid_content = [id_old2new[x] for x in valid_content]

            # count_vector = np.zeros([chi_size])
            # for id in valid_content:
            #     count_vector[id] += 1
            # tfidf_list = np.multiply(count_vector,weights)
            # x_data[i,:] = tfidf_list

            id_counter = Counter(valid_content)
            for id in id_counter.keys():
                count = id_counter[id]
                weight = count * weights[id]
                # x_data[i, id] = count
                if valid_info_count<train_num:
                    rows.append(valid_info_count)
                    cols.append(id)
                    data.append(weight)
                else:
                    test_rows.append(valid_info_count-train_num)
                    test_cols.append(id)
                    test_data.append(weight)
            if valid_info_count<train_num:
                out_list.append(label_list.index(label))
            else:
                test_out_list.append(label_list.index(label))
            valid_info_count += 1

        x_data = sparse.coo_matrix((data,(rows,cols)), shape=(train_num, chi_size))
        x_test_data = sparse.coo_matrix((test_data,(test_rows, test_cols)), shape=(valid_info_count-train_num,chi_size))
        print(x_data.shape, len(out_list))
        print(x_test_data.shape, len(test_out_list))
        return x_data, out_list, x_test_data, test_out_list


def train(train_input, train_output, test_input, test_output):
    # 训练模块

    # 选择模型
    # model = MultinomialNB()
    # model = GaussianNB()
    model = SGDClassifier()
    # model = SVC(kernel='linear')
    # model = LinearSVC()
    # model = RandomForestClassifier(max_depth=2, n_estimators=500)
    # model = AdaBoostClassifier(n_estimators=500,base_estimator=DecisionTreeClassifier(max_depth=10))

    # 训练 & 评测
    model.fit(train_input,train_output)
    pred_train = model.predict(train_input)
    pred_test = model.predict(test_input)

    label_size = max(train_output)+1
    train_ratio = cal_accuracy(pred_train, train_output)
    train_recal = cal_recall(pred_train, train_output, label_size)
    print(test_output)
    print(list(pred_test))
    test_ratio = cal_accuracy(pred_test, test_output)
    test_recal = cal_recall(pred_test, test_output, label_size)
    print('%f\t%f'%(train_ratio, test_ratio))
    print('%f\t%f'%(train_recal, test_recal ))

def cal_accuracy(pred, output):
    assert len(pred) == len(output)
    size = len(pred)
    res = [1 if pred[i]==output[i] else 0 for i in range(size)]
    accu_ratio = sum(res) / len(res)
    return accu_ratio

def cal_recall(pred, output, label_size):
    pred_list = np.zeros([label_size])
    accu_list = np.zeros([label_size])
    for i in range(len(pred)):
        pred_list[pred[i]] += 1
        if pred[i]==output[i]:
            accu_list[pred[i]] += 1
    recal_list = np.true_divide(accu_list, pred_list)
    return np.mean(recal_list)

if __name__=='__main__':
    train_in, train_out, test_in, test_out = gen_data_sparse()

    train(train_in, train_out, test_in, test_out)


