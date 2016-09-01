import os
# from THUNewsClassify.gensim_edition import loadFiles
from gensim import corpora, models
import jieba
import re
import matplotlib.pyplot as plt
from multiprocessing import Process,Queue,Lock
import time
import itertools

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

def convert_doc_to_wordlist(str_doc,cut_all):
    sent_list = str_doc.split('\n')
    sent_list = map(rm_char, sent_list) # 去掉一些字符，例如\u3000
    word_2dlist = [rm_tokens(jieba.cut(part,cut_all=cut_all)) for part in sent_list] # 分词
    word_list = sum(word_2dlist,[])
    return word_list

def generate_dict_subprocess(id,
                             p_num,
                             dict_queue,
                             cut_all,
                             file_parent_path='/mnt/D/multiangle/DataSet/THUCNews'
                             ):
    files = loadFiles(file_parent_path)
    dictionary = corpora.Dictionary()
    file_count = 0
    exe_count = 0
    for file in files:
        file_count += 1
        if file_count%p_num==id:
            exe_count += 1
            file = convert_doc_to_wordlist(file, cut_all)
            dictionary.add_documents([file])
            if exe_count%100==0:
                print('Process {i} has execute {c} at {t}'.format(
                    c=exe_count,
                    t=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()),
                    i=id)
                )
                print(file[:min(file.__len__(),5)])
    dict_queue.put(dictionary)

def genDict(path_parent, path_dict_folder):
    # 第一次遍历，成立词典，获取词频，文频等信息
    p_pool = []
    dict_queue = Queue()
    p_num = 3
    for i in range(p_num):
        p = Process(target=generate_dict_subprocess,
                    args=(i,
                          p_num,
                          dict_queue,
                          cut_all,
                          # '/mnt/D/multiangle/DataSet/THUCNews'
                          path_parent
                          ))
        p_pool.append(p)

    for p in p_pool: # 启动进程
        # p = Process(p)
        p.start()

    while True: # 检测是否全部完成
        if dict_queue.qsize() >= p_num:
            break
        time.sleep(1)

    dictionary = corpora.Dictionary()
    for i in range(p_num):
        q_dict = dict_queue.get()
        dictionary.merge_with(q_dict)

    for p in p_pool:
        p.terminate()

    dictionary.save(os.path.join(path_dict_folder,'THUNews.dict'))
    dictionary.save_as_text(os.path.join(path_dict_folder,'THUNews.txt'))

def convDoc2Vector(path_doc_parent,path_dict_folder,path_root):
    dictionary = corpora.Dictionary.load(os.path.join(path_dict_folder,'THUNews_picked.dict'))
    print(os.path.join(path_dict_folder,'THUNews_picked.dict'))
    for folder in loadFolders(path_doc_parent):
        folder_name = folder.split('/')[-1]
        print(folder_name)
        files = os.listdir(folder)
        cate_bow = []
        count = 0
        for file in files:
            count += 1
            if count%100 == 0 :
                print('{c} at {t}'.format(c=count, t=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())))
            if count%10 > 0: # 抽样 n抽1
                continue
            file_path = os.path.join(folder,file)
            file = open(file_path,'rb')
            doc = file.read().decode('utf8')
            word_list = convert_doc_to_wordlist(doc, cut_all)
            word_bow = dictionary.doc2bow(word_list)
            cate_bow.append(word_bow)
            file.close()

        tmp_path = os.path.join(path_root,'bow_sampling')
        if not os.path.exists(tmp_path):
            os.mkdir(tmp_path)
        corpora.MmCorpus.serialize(tmp_path+'/{x}.mm'.format(x=folder_name),
                                   cate_bow,
                                   id2word=dictionary,
                                   # labels=folder_name
                                   )

if __name__=='__main__':

    cut_all = True # 是否要把所有可能的单词都列出来？ true 表示是 , false 表示否
    path_root   = '/media/multiangle/F/DataSet/THUCNews'
    path_doc_parent = os.path.join(path_root,'THUCNewsTotal')
    os.chdir(path_doc_parent)
    path_dict_folder = os.path.join(path_root, 'THUNewsDict') # 存放词典的地方
    if not os.path.exists(path_dict_folder):
        os.mkdir(path_dict_folder)


    # # ===================================================================
    # # 第一次遍历，成立词典，获取词频，文频等信息
    # genDict(path_doc_parent, path_dict_folder)

    # # ===================================================================
    # # 去掉词典中出现次数过少的
    # dictionary = corpora.Dictionary.load(os.path.join(path_dict_folder, 'THUNews.dict'))
    # small_freq_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq < 5 ]
    # dictionary.filter_tokens(small_freq_ids)
    # dictionary.compactify()
    # dictionary.save(os.path.join(path_dict_folder, 'THUNews_picked.dict'))

    # # ===================================================================
    # # 第二次遍历，开始将文档转化成id稀疏表示
    # convDoc2Vector(path_doc_parent, path_dict_folder, path_root)

    # # # ===================================================================
    # # 第三次遍历，开始将文档转化成tf idf 表示
    # dictionary = corpora.Dictionary.load(os.path.join(path_dict_folder,'THUNews_picked.dict'))
    # bow_path = path_root + '/bow_sampling'
    # tfidf_path = path_root + '/tfidf_sampling'
    # if not os.path.exists(tfidf_path):
    #     os.mkdir(tfidf_path)
    # files = os.listdir(bow_path)
    # cate_set = set([x.split('.')[0] for x in files])
    # for cat in cate_set:
    #     path = '{pp}/{cat}.mm'.format(pp=bow_path, cat=cat)
    #     corpus = corpora.MmCorpus(path)
    #     tfidf_model = models.TfidfModel(corpus=corpus,
    #                                     dictionary=dictionary)
    #     corpus_tfidf = [tfidf_model[doc] for doc in corpus]
    #     corpora.MmCorpus.serialize('{f}/{c}.mm'.format(f=tfidf_path,c=cat),
    #                                corpus_tfidf,
    #                                id2word=dictionary
    #                                )
    #     print('{f}/{c}.mm'.format(f=tfidf_path,c=cat))

    # # # ===================================================================
    # # 第四次遍历，计算lsi
    dictionary  = corpora.Dictionary.load(os.path.join(path_dict_folder,'THUNews_picked.dict'))
    tfidf_path  = path_root + '/tfidf_sampling'
    lsi_path    = path_root + '/lsi_sampling'
    if not os.path.exists(lsi_path):
        os.mkdir(lsi_path)
    files = os.listdir(tfidf_path)
    cate_list = list(set([x.split('.')[0] for x in files]))
    doc_num_list = []
    tfidf_corpus_total = None
    for cat in cate_list:
        path = '{pp}/{cat}.mm'.format(pp=tfidf_path, cat=cat)
        corpus = corpora.MmCorpus(path)
        doc_num_list.append(corpus.num_docs)
        if not tfidf_corpus_total:
            tfidf_corpus_total = [x for x in corpus]
        else:
            tfidf_corpus_total += [x for x in corpus]
        print('category {c} loaded,len {l} at {t}'
              .format(c=cat,l=corpus.num_docs,t=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())))
    lsi_model = models.LsiModel(corpus = tfidf_corpus_total, id2word = dictionary, num_topics=10)
    print('lsi model is generated at {t}'.format(t=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())))
    del tfidf_corpus_total  # 总共的tfidf corpus已经用完，释放变量空间
    for cat in cate_list:
        path = '{pp}/{cat}.mm'.format(pp=tfidf_path, cat=cat)
        corpus = corpora.MmCorpus(path)
        corpus_lsi = [lsi_model[doc] for doc in corpus]
        corpora.MmCorpus.serialize('{f}/{c}.mm'.format(f=lsi_path,c=cat),
                                   corpus_lsi,
                                   id2word=dictionary
                                   )
        print('category {c} generate lsi vector, at {t}'
              .format(c=cat,t=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())))













