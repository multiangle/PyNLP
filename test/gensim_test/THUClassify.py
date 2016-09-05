from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os
from gensim import corpora, models
import time
from scipy.sparse import csr_matrix
from sklearn import svm
import logging
import pickle as pkl
import numpy as np
import jieba
from pprint import pprint

# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s %(levelname)s %(message)s')



path_root   = '/media/multiangle/F/DataSet/THUCNews'
path_dict_folder = os.path.join(path_root, 'THUNewsDict') # 存放词典的地方
dictionary  = corpora.Dictionary.load(os.path.join(path_dict_folder,'THUNews_picked.dict'))
path_tmp    = path_root + '/tmp'
#
lsi_path    = path_tmp + '/lsi_sampling'
files       = os.listdir(lsi_path)
cate_list       = list(set([x.split('.')[0] for x in files]))
cate_path_list  = [lsi_path + '/' + cat for cat in cate_list]

# doc_num_list = []
# tag_list    = []
# lsi_corpus_total = None
# count = 0
# for cat in cate_list:
#     path = '{pp}/{cat}.mm'.format(pp=lsi_path, cat=cat)
#     corpus = corpora.MmCorpus(path)
#     doc_num_list.append(corpus.num_docs)
#     tag_list += [count]*corpus.num_docs
#     count += 1
#     if not lsi_corpus_total:
#         lsi_corpus_total = [x for x in corpus]
#     else:
#         lsi_corpus_total += [x for x in corpus]
#     print('category {c} loaded,len {l} at {t}'
#           .format(c=cat,l=corpus.num_docs,t=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())))
#
# # 将gensim中的mm表示转化成numpy矩阵表示
# data = []
# rows = []
# cols = []
# line_count = 0
# for line in lsi_corpus_total:
#     for elem in line:
#         rows.append(line_count)
#         cols.append(elem[0])
#         data.append(elem[1])
#     line_count += 1
# lsi_matrix = csr_matrix((data,(rows,cols))).toarray()
# rarray=np.random.random(size=line_count)
# train_set = []
# train_tag = []
# test_set = []
# test_tag = []
# for i in range(line_count):
#     if rarray[i]<0.8:
#         train_set.append(lsi_matrix[i,:])
#         train_tag.append(tag_list[i])
#     else:
#         test_set.append(lsi_matrix[i,:])
#         test_tag.append(tag_list[i])
# svm_classify(lsi_matrix,tag_list)

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

    x = open(+'/svm_classifier.pkl','wb')
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

def gen_lsi_vector(content, path_dict_folder):
    content = [i for i in jieba.cut(content,cut_all=False)]
    print(content)
    dictionary = corpora.Dictionary.load(os.path.join(path_dict_folder,'THUNews_picked.dict'))
    corpus = dictionary.doc2bow(content)
    corpus = [corpus]
    print(corpus)
    tfidf_model = models.TfidfModel(corpus=corpus,
                                    dictionary=dictionary)
    corpus_tfidf = [tfidf_model[doc] for doc in corpus]
    print(corpus_tfidf)
    path_root   = '/media/multiangle/F/DataSet/THUCNews'
    path_tmp    = path_root + '/tmp'
    file = open(path_tmp+'/lsi_model.pkl','rb')
    lsi_model = pkl.load(file)
    file.close()
    # lsi_model = models.LsiModel(corpus = corpus_tfidf, id2word = dictionary, num_topics=50)
    corpus_lsi = [lsi_model[doc] for doc in corpus]
    print(corpus_lsi)
    data = []
    rows = []
    cols = []
    corpus_lsi = corpus_lsi[0]
    for item in corpus_lsi:
        data.append(item[1])
        rows.append(0)
        cols.append(item[0])
    content = csr_matrix((data,(rows,cols))).toarray()
    print(content)

    file = open(path_root+'/svm_classifier.pkl','rb')
    svm_classifier = pkl.load(file)
    cat = svm_classifier.predict(content)
    print(cat)
    print(cate_list[cat])


path_root   = '/media/multiangle/F/DataSet/THUCNews'
path_dict_folder = os.path.join(path_root, 'THUNewsDict') # 存放词典的地方
content = """
第三、苏南怎么了
首先，经历了高增长之后，势必有一个低增长期，经济发展的必然，转型也需要时间，没有什么大惊小怪。
其次，大环境是外资企业经营困难或者撤离，本质上就是一件自然的发展规律，也是转型升级的一个过程。
再次，政府粗暴干涉经济....
这是一个悲桑的故事。典型是无锡。无锡太湖蓝藻事件之后，或许是政绩关系，时无锡书记制定了一揽子计划，其中一项就是把无锡境内的大大小小涉及环保的工厂，全部关闭，呵呵哒。所以后来在安徽郎溪有了无锡工业园(当年滨湖区华庄一个镇就关了1000多家企业！！！！郎溪无锡工业园，至今还是安徽宣州招商的政绩骄傲。为毛无锡人去郎溪？因为近啊，走宜兴方向才1个多小时，比去苏北近多了，早上去郎溪晚上回无锡，1小时就当堵车了)。就是不管三七二十一，我不管你环保达不达标，只要有可能产生污染，全部关门...然后，以政府力量，推行了530，尚德等一系列“产品”，后来全部失败，还有很多槽点。
无锡之后就被南京超越。（TM一个堂堂江宁府，和常州府三个县城，600万人，有毛可比性）
今天看来，实在可笑。
太湖蓝藻至今还有，因为这不单单是无锡的事情，夏天一刮东南风，整个太湖的脏东西都往无锡飘，单单关了无锡的工厂，有毛线用。530引来的基本全是骗子（当然也有成果）。至于尚德，呵呵呵。
无锡另一个特色产业是电动车，没错，电动小毛驴，当时的无锡市政府嫌LOW，极力压制，电动车企业纷纷到广东、湖南一代买地建生产基地....现在呢，纷纷要求电动车企业回来，我给你土地给你政策给你优惠，你倒是给我搬回来啊！小刀，艾玛，新日等等一大片大型电动车生产企业都在无锡…
所以被超千亿，根本不是什么大事，可能就一群企业的量.....而这些企业全被赶跑了！
无锡是个典型的苏南城市，也是一个典型的工业城市。确实遇到发展瓶颈，需要转型，但是时无锡领导粗暴的以政府力量干涉经济转型，结果很明显。脱离了乡镇经济去搞高大上，不是空中楼阁么？接着来了一个啥也不干、专门出席各种典礼的阿姨，坐实了无锡的败局。
但无锡乙烷了吗？只要乡镇企业还存在，无锡就没完，请别忽视了草根的力量。
苏州形式比无锡要好，苏州没有无锡这样的折腾，特别是苏州发展了金融业，还是比较高大上的。但是产业转型，和无锡一样，依旧在摸索中。
"""
gen_lsi_vector(content, path_dict_folder)
