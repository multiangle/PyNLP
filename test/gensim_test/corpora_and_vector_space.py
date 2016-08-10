
import logging
from gensim import corpora
import re
import jieba
from collections import defaultdict
from pprint import pprint
from pprint import pprint  # pretty-printer


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

# remove common words and tokenize
# 去掉停用词
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]

# remove words that appear only once
# 去掉只出现一次的单词
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
texts = [[token for token in text if frequency[token] > 1]
         for text in texts]


# 将文档存入字典，字典有很多功能，比如
# diction.token2id 存放的是单词-id key-value对
# diction.dfs 存放的是单词的出现频率
dictionary = corpora.Dictionary(texts)
# ditionary 还可以一篇篇的加文档 dictionary.add_documents(args) 注意args是[[],[],...]格式
# 还可以filter功能。 dictionary.filter_tokens()
# dictionary.compactify()  # remove gaps in id sequence after words that were removed
dictionary.save('/tmp/deerwester.dict')  # store the dictionary, for future reference

# 输出dictionary中个单词的出现频率
def PrintDictionary():
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

# 测试 ditonary的doc2bow功能，转化为one-hot presentation
new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec)  # the word "interaction" does not appear in the dictionary and is ignored

# 将文本转化为 doc2bow 形式的数组
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)  # store to disk, for later use
# 除了MmCorpus以外，还有SvmLightCorpus等以各种格式存入磁盘




