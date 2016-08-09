
import logging
from gensim import corpora
import re
import jieba
from collections import defaultdict
from pprint import pprint

def etl(s): #remove 标点和特殊字符
    s = re.sub(r'\u3000','',s)
    s = re.sub(r' ','',s)
    return s

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

document = """北京体育局辟谣没收到张怡宁退役申请 太不符逻辑
　　本报讯(记者 崔方舟)刚刚在新闻发布会上说暂不退役的张怡宁，近日又“被爆出”已经递交退役申请的消息。昨日，北京市体育局竞技处负责人孙国华出面否定了这一说法，“我们从来就没有收到过张怡宁的退役申请。”
　　对于前天马来西亚中文媒体表示张怡宁已经递交退役申请的报道，孙国华直言“不可能”。对于这样的消息孙国华也表示非常无奈，“前几天张怡宁才开了新闻发布会说不退役，她不可能刚说完不退役就递交申请吧，想想也知道不符合逻辑。”据悉，按照正常程序，由于张怡宁的档案关系仍挂在先农坛体校，如果她要退役，首先要向体校递交退役申请。先农坛体校校长孟强华也表示，自全运会结束后张怡宁从未向队中递交退役申请，而北京队教练张雷也称没有听说张怡宁向队中申请退役。
""".split('\n')
document = list(map(etl,document))

# remove stop words
stop_list = set('( ) 一 太'.split(' '))
texts = [[word for word in jieba.cut(etl(line), cut_all=True) if word not in stop_list]
         for line in document]

# remove words appears only once
tokens = sum(texts,[])

# token_once = [word for word in set(tokens) if tokens.count(word)<2] # using set to pick out the once words
# texts_gt_once = [[word for word in line if word not in token_once] for line in texts]

counter = defaultdict(int)  # use defaultdict to pick out the once words
for word in tokens:
    counter[word] += 1
texts = [[word for word in line if counter[word]>1] for line in texts]

dictionary = corpora.Dictionary(texts)

# # 测试讲文档转化为向量
# test_doc = "张怡宁表示即将退役"    # 讲文档转化成向量，不过是one-hot presentation,稀疏表示
# test_doc = jieba.cut(test_doc,cut_all=False)
# test_vec = dictionary.doc2bow(test_doc)

corpus = [dictionary.doc2bow(line) for line in texts]
pprint(corpus)




