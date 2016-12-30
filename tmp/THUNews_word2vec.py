import tensorflow as tf
import numpy as np
from embedding.word2vec import  NEGModel
import os
from pprint import pprint
import random
import jieba

def rm_chars(sent):
    new_sent = ''
    warn_chars = [
        '\u3000',
        '\n',
        '\xa0',
        '□',
        '■',
        '●',
    ]
    for char in sent:
        if char in warn_chars:
            continue
        new_sent += char
    return new_sent

def read_text(file_path):
    lines = []
    with open(file_path) as f:
        line = f.readline()
        while line:
            dealed_line = rm_chars(line)
            if dealed_line.__len__()>0:
                lines.append(dealed_line)
            line = f.readline()
        return lines

if __name__=='__main__':
    # 生成所有文件列表
    data_root_path = '/media/multiangle/F/DataSet/THUCNews/THUCNewsPart'
    cate_list = os.listdir(data_root_path)
    file_info_list = []
    for cate in cate_list:
        tmp_path = os.path.join(data_root_path,cate)
        files = os.listdir(tmp_path)
        for file_path in files:
            abs_path = os.path.join(tmp_path,file_path)
            tmp_info = {}
            tmp_info['path'] = abs_path
            tmp_info['label'] = cate
            # tmp_info['size'] = os.path.getsize(abs_path)
            file_info_list.append(tmp_info)
    random.shuffle(file_info_list)

    # 统计单词，生成词典
    word_list = []
    word_dict = {}
    for file_info in file_info_list:
        tmp_path = file_info['path']
        lines = read_text(tmp_path)
        context = "".join(lines)



