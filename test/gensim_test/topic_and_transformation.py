import os
from gensim import corpora, models, similarities
from pprint import pprint
from matplotlib import pyplot as plt
import logging

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def PrintDictionary(dictionary):
    token2id = dictionary.token2id
    dfs = dictionary.dfs
    token_info = {}
    for word in token2id:
        token_info[word] = dict(
            word = word,
            id = token2id[word],
            freq = dfs[token2id[word]]
        )
    token_items = token_info.values()
    token_items = sorted(token_items, key = lambda x:x['id'])
    print('The info of dictionary: ')
    pprint(token_items)
    print('--------------------------')

def Show2dCorpora(corpus):
    nodes = list(corpus)
    ax0 = [x[0][1] for x in nodes] # 绘制各个doc代表的点
    ax1 = [x[1][1] for x in nodes]
    # print(ax0)
    # print(ax1)
    plt.plot(ax0,ax1,'o')
    plt.show()

if (os.path.exists("/tmp/deerwester.dict")):
    dictionary = corpora.Dictionary.load('/tmp/deerwester.dict')
    corpus = corpora.MmCorpus('/tmp/deerwester.mm')
    print("Used files generated from first tutorial")
else:
    print("Please run first tutorial to generate data set")

PrintDictionary(dictionary)

# 尝试将corpus(bow形式) 转化成tf-idf形式
tfidf_model = models.TfidfModel(corpus) # step 1 -- initialize a model 将文档由按照词频表示 转变为按照tf-idf格式表示
doc_bow = [(0, 1), (1, 1),[4,3]]
doc_tfidf = tfidf_model[doc_bow]

# 将整个corpus转为tf-idf格式
corpus_tfidf = tfidf_model[corpus]
# pprint(list(corpus_tfidf))
# pprint(list(corpus))

## LSI模型 **************************************************
# 转化为lsi模型, 可用作聚类或分类
lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
corpus_lsi = lsi_model[corpus_tfidf]
nodes = list(corpus_lsi)
# pprint(nodes)
lsi_model.print_topics(2) # 打印各topic的含义

# ax0 = [x[0][1] for x in nodes] # 绘制各个doc代表的点
# ax1 = [x[1][1] for x in nodes]
# print(ax0)
# print(ax1)
# plt.plot(ax0,ax1,'o')
# plt.show()

lsi_model.save('/tmp/model.lsi') # same for tfidf, lda, ...
lsi_model = models.LsiModel.load('/tmp/model.lsi')
#  *********************************************************

## LDA模型 **************************************************
lda_model = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=2)
corpus_lda = lsi_model[corpus_tfidf]
Show2dCorpora(corpus_lda)
# nodes = list(corpus_lda)
# pprint(list(corpus_lda))

# 此外，还有Random Projections, Hierarchical Dirichlet Process等模型
