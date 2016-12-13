from gensim import corpora
from pymongo import MongoClient

client = MongoClient('localhost',27017)
db = client.microblog_classify
collect = db['test']
cursor = collect.find()
for line in cursor:
    print(line)