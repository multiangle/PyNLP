
from pymongo import MongoClient
import jieba
import TextDeal
from collections import Counter

def readFile(file_path):
    res = []
    with open(file_path,'r') as f:
        line = f.readline()
        while line:
            dealed = line[:-1]
            res.append(dealed)
            line = f.readline()
    return res

pos_words = set(readFile('../../static/positive_words.txt'))
neg_words = set(readFile('../../static/negtive_words.txt'))
pos_words.remove('')
neg_words.remove('')

client = MongoClient('localhost',27017)
db = client['microblog_spider']
collection = db['user_2016_12']
handle = collection.find().limit(5000)
for i,line in enumerate(handle):
    ori_text = line['ori_text']
    ori_text = TextDeal.removeLinkOnly(ori_text)
    # ori_text = TextDeal.removeNetworkLinkOnly(ori_text)
    ori_text = TextDeal.removeLinkTotal(ori_text)
    ori_text = TextDeal.removeIBlockTotal(ori_text)
    words = list(jieba.cut(ori_text,cut_all='false'))
    word_counter = Counter(words)
    pos_v = 0
    neg_v = 0
    for word in word_counter:
        if word in pos_words:
            pos_v += word_counter[word]
        if word in neg_words:
            neg_v += word_counter[word]
    # print(ori_text+'+++')
    emo_index = (pos_v - neg_v)/(len(words)+1)
    if (abs(pos_v-neg_v)>2):
        print('{v}\t{t}\t++'.format(v=emo_index,t=ori_text))
    # print('{a}\t{b}\t{c}\t{d}\t--'.format(a=pos_v,b=neg_v,c=pos_v-neg_v,d=emo_index))
    # word_counter.pop('')
client.close()
