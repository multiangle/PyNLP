from gensim import corpora, models, similarities
from pprint import pprint
from matplotlib import pyplot as plt

def Show2dCorpora(corpus):
    nodes = list(corpus)
    ax0 = [x[0][1] for x in nodes] # 绘制各个doc代表的点
    ax1 = [x[1][1] for x in nodes]
    # print(ax0)
    # print(ax1)
    plt.plot(ax0,ax1,'o')
    # plt.show()
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

# 读取词典和文档，计算tf idf 和 lsi 的表达方式，并生成相似度矩阵
dictionary = corpora.Dictionary.load('/tmp/deerwester.dict')
corpus = corpora.MmCorpus('/tmp/deerwester.mm') # comes from the first tutorial, "From strings to vectors"
tfidf_model = models.TfidfModel(corpus)
corpus_tfidf = tfidf_model[corpus]
lsi_model = models.LsiModel(corpus_tfidf,id2word=dictionary,num_topics=2)
corpus_lsi = lsi_model[corpus_tfidf]
corpus_simi_matrix = similarities.MatrixSimilarity(corpus_lsi)
corpus_simi_matrix.save('/tmp/deerwester.index')
similarities.MatrixSimilarity.load('/tmp/deerwester.index')

# 计算一个新的文本与既有文本的相关度
test_text = "Human computer interaction".split()
test_bow = dictionary.doc2bow(test_text)
test_tfidf = tfidf_model[test_bow]
test_lsi = lsi_model[test_tfidf]
test_simi = corpus_simi_matrix[test_lsi]
print(list(enumerate(test_simi)))