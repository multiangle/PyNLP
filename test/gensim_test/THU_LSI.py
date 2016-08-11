import os
# from THUNewsClassify.gensim_edition import loadFiles
from gensim import corpora, models
import jieba
import re
import matplotlib.pyplot as plt
from multiprocessing import Process
import time

class loadFolders(object):
    def __init__(self,par_path):
        self.par_path = par_path
    def __iter__(self):
        for file in os.listdir(self.par_path):
            file_abspath = os.path.join(self.par_path, file)
            if os.path.isdir(file_abspath): # if file is a folder
                yield file_abspath

class loadFiles(object):
    def __init__(self,par_path):
        self.par_path = par_path
    def __iter__(self):
        folders = loadFolders(self.par_path)
        for folder in folders:              # level directory
            for file in os.listdir(folder):     # secondary directory
                file_path = os.path.join(folder,file)
                if os.path.isfile(file_path):
                    this_file = open(file_path,'rb')
                    content = this_file.read().decode('utf8')
                    yield content
                    this_file.close()

def rm_char(text):
    text = re.sub('\u3000','',text)
    return text

def get_stop_words(path='/home/multiangle/coding/python/PyNLP/static/stop_words.txt'):
    file = open(path,'rb').read().decode('utf8').split('\n')
    return set(file)

def rm_tokens(words): # 去掉一些停用次和数字
    words_list = list(words)
    stop_words = get_stop_words()
    for i in range(words_list.__len__())[::-1]:
        if words_list[i] in stop_words: # 去除停用词
            words_list.pop(i)
        elif words_list[i].isdigit():
            words_list.pop(i)
    return words_list

def generate_dict_subprocess(id,p_num,dictionary,files,cut_all):
    file_count = 0
    exe_count = 0
    for file in files:
        file_count += 1
        if file_count%p_num==id:
            exe_count += 1
            file = file.split('\n')
            file = map(rm_char, file) # 去掉一些字符，例如\u3000
            file = [rm_tokens(jieba.cut(part,cut_all=cut_all)) for part in file] # 分词
            file = sum(file,[])
            dictionary.add_documents([file])
            if exe_count%100==0:
                print('Process {i} has execute {c} at {t}'.format(
                    c=exe_count,
                    t=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()),
                    i=id)
                )
                print(file[:min(file.__len__(),5)])

if __name__=='__main__':

    cut_all = True # 是否要把所有可能的单词都列出来？ true 表示是 , false 表示否

    # 第一次遍历，成立词典，获取词频，文频等信息
    dictionary = corpora.Dictionary()
    p_pool = []
    dict_pool = []
    p_num = 3
    for i in range(p_num):
        files = loadFiles('/mnt/D/multiangle/DataSet/THUCNews')
        d = corpora.Dictionary()
        dict_pool.append(d)
        p = Process(target=generate_dict_subprocess,
                                    args=(i, p_num, d, files, cut_all))
        p_pool.append(p)

    for p in p_pool: # 启动进程
        p.start()

    while True: # 检测是否全部完成
        all_finished = True
        for p in p_pool:
            if p.is_alive():
                all_finished = False
        if all_finished:
            break
        time.sleep(1)

    dictionary = corpora.Dictionary()
    for dic in dict_pool:
        dictionary.merge_with(dic)
    dictionary.save('./THUNews_multiprocess.dict')
    # dictionary.save_as_text('./THUNews_dict.txt')







