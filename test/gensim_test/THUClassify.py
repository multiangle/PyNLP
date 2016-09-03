from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os
from gensim import corpora, models
import time
from scipy.sparse import csr_matrix
from sklearn import svm
import logging
import pickle as pkl
import numpy as np

# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s %(levelname)s %(message)s')

path_root   = '/media/multiangle/F/DataSet/THUCNews'
path_dict_folder = os.path.join(path_root, 'THUNewsDict') # 存放词典的地方
dictionary  = corpora.Dictionary.load(os.path.join(path_dict_folder,'THUNews_picked.dict'))
path_root   = '/media/multiangle/F/DataSet/THUCNews'

lsi_path    = path_root + '/lsi_sampling'
files       = os.listdir(lsi_path)
cate_list       = list(set([x.split('.')[0] for x in files]))
cate_path_list  = [lsi_path + '/' + cat for cat in cate_list]

doc_num_list = []
tag_list    = []
lsi_corpus_total = None
count = 0
for cat in cate_list:
    path = '{pp}/{cat}.mm'.format(pp=lsi_path, cat=cat)
    corpus = corpora.MmCorpus(path)
    doc_num_list.append(corpus.num_docs)
    tag_list += [count]*corpus.num_docs
    count += 1
    if not lsi_corpus_total:
        lsi_corpus_total = [x for x in corpus]
    else:
        lsi_corpus_total += [x for x in corpus]
    print('category {c} loaded,len {l} at {t}'
          .format(c=cat,l=corpus.num_docs,t=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())))

# 将gensim中的mm表示转化成numpy矩阵表示
data = []
rows = []
cols = []
line_count = 0
for line in lsi_corpus_total:
    for elem in line:
        rows.append(line_count)
        cols.append(elem[0])
        data.append(elem[1])
    line_count += 1
lsi_matrix = csr_matrix((data,(rows,cols))).toarray()
rarray=np.random.random(size=line_count)
train_set = []
train_tag = []
test_set = []
test_tag = []
for i in range(line_count):
    if rarray[i]<0.8:
        train_set.append(lsi_matrix[i,:])
        train_tag.append(tag_list[i])
    else:
        test_set.append(lsi_matrix[i,:])
        test_tag.append(tag_list[i])


# 测试线性分类
def linear_classify(data,tag):
    lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    lda_res = lda.fit(train_set, train_tag)
    train_pred  = lda_res.predict(train_set)
    test_pred = lda_res.predict(test_set)
    x = open(path_root+'/linear_classifier.pkl','wb')
    pkl.dump(lda_res, x)
    x.close()
    train_err_num, train_err_ratio = checkPred(train_tag, train_pred)
    test_err_num, test_err_ratio  = checkPred(test_tag, test_pred)

    print(train_err_num)
    print(train_err_ratio)
    print(test_err_num)
    print(test_err_ratio)

def svm_classify(data, tag):

    clf = svm.LinearSVC()
    clf_res = clf.fit(train_set,train_tag)
    train_pred  = clf_res.predict(train_set)
    test_pred = clf_res.predict(test_set)

    train_err_num, train_err_ratio = checkPred(train_tag, train_pred)
    test_err_num, test_err_ratio  = checkPred(test_tag, test_pred)

    print(train_err_num)
    print(train_err_ratio)
    print(test_err_num)
    print(test_err_ratio)

    x = open(path_root+'/svm_classifier.pkl','wb')
    pkl.dump(clf, x)

# cal the result of prediction
def checkPred(data_tag, data_pred):
    if data_tag.__len__() != data_pred.__len__():
        raise RuntimeError('The length of data tag and data pred should be the same')
    err_count = 0
    for i in range(data_tag.__len__()):
        if data_tag[i]!=data_pred[i]:
            err_count += 1
    err_ratio = err_count / data_tag.__len__()
    return [err_count, err_ratio]

svm_classify(lsi_matrix,tag_list)

