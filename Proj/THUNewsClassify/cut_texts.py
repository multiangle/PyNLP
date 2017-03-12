import numpy as np
import tensorflow as tf
import pickle as pkl
from Proj.THUNewsClassify.util import pick_valid_word,read_text,rm_words
import jieba
from pprint import pprint

with open('file_info_list.pkl','rb') as f:
    file_info_list = pkl.load(f)
with open('word_list_path.pkl','rb') as f:
    word_info_list = pkl.load(f)
    # 包含了 count, word, id, sub_count这几项
full_word_info_dict = {}
for line in word_info_list:
    full_word_info_dict[line['word']] = line
full_segment_res = []
for count,file_info in enumerate(file_info_list):
    path = file_info['path']
    label = file_info['label']
    lines = read_text(path)
    info = {}
    line_as_id = []
    for line in lines:
        ids = []
        res = list(jieba.cut(line,cut_all=False))
        valid_words = rm_words(res)
        for word in valid_words:
            if word in full_word_info_dict:
                ids.append(full_word_info_dict[word]['id'])
        if len(ids)>0:
            line_as_id.append(ids)
    if len(line_as_id)>0:
        info['content'] = line_as_id
        info['path'] = path
        info['label'] = label
        full_segment_res.append(info)
    if count%1000==0:
        print(count,'个文件分割完毕')
with open('THUCNews_fullid_type.pkl','wb') as f:
    pkl.dump(full_segment_res,f)

