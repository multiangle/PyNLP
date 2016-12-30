
import re
import os

global stop_words_list
global stop_words_set
stop_words_list = None
stop_words_set = None
invalid_word_list = None
invalid_word_set = None

def removeLinkOnly(sentence):
    # 去掉 a 块部分
    m1 = re.compile('<a.*?/a>')
    res = re.findall(m1,sentence)
    if res.__len__()>0:
        for item in res:
            m2 = re.compile('#(.*?)#')
            v = re.findall(m2,item)
            if v.__len__()==1:
                sentence = sentence.replace(item,v[0])

    # 去掉<br/>
    while '<br/>' in sentence:
        sentence = sentence.replace('<br/>','')

    return sentence

def isStopWord(word):
    global stop_words_list
    global stop_words_set
    if stop_words_list==None:
        stop_words_list = []
        abs_path = '/home/multiangle/coding/python/PyNLP/static/stop_words.txt'
        with open(abs_path) as f:
            line = f.readline()
            while line:
                stop_words_list.append(line[:-1])
                line = f.readline()
        stop_words_set = set(stop_words_list)
    return stop_words_set.__contains__(word)

def isValidWord(word):
    global invalid_word_list
    global invalid_word_set
    if invalid_word_list == None:
        invalid_word_list = []
        abs_path = '/home/multiangle/coding/python/PyNLP/static/invalid_words.txt'
        with open(abs_path) as f:
            line = f.readline()
            while line:
                invalid_word_list.append(line[:-1])
                line = f.readline()
        invalid_word_set = set(invalid_word_list)
    return not invalid_word_set.__contains__(word)

if __name__=='__main__':
    # print(isStopWord('23333'))
    print(isValidWord(""))