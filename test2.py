from gensim import corpora
from pymongo import MongoClient
from DB_Interface import MySQL_Interface
from matplotlib import pyplot as plt
from pprint import pprint
import pickle as pkl
import math

# mysql = MySQL_Interface(dbname='milu')
#
# content = mysql.select_asQuery("select * from weibo_hadoop_wordcount")
# col_info = mysql.get_col_name('weibo_hadoop_wordcount')
#
# word_info = {}
# for line in content:
#     word = line[0]
#     item_info = word_info.get(word,{})
#     if item_info == {} :
#         item_info['word'] = word
#         item_info['freq'] = line[2]
#         item_info['reposts'] = line[3]
#         item_info['attitudes'] = line[4]
#         item_info['comments'] = line[5]
#     else:
#         item_info['freq'] += line[2]
#         item_info['reposts'] += line[3]
#         item_info['attitudes'] += line[4]
#         item_info['comments'] += line[5]
#     word_info[word] = item_info
#
# with open('word_count','wb') as f:
#     pkl.dump(word_info,f)

content = None
with open('word_count','rb') as f:
    content = pkl.load(f)

cont_list = [content[x] for x in content]
cont_list.sort(key=lambda x:x['freq'],reverse=True)
pprint(cont_list[:40000])
