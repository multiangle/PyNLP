
import TextDeal
from Proj.THUNewsClassify.chi_square import ChiSquareCalculator
import numpy as np

def rm_chars(sent):
    new_sent = ''
    warn_chars = [
        '\u3000',
        '\n',
        '\xa0',
        '□',
        '■',
        '●',
        '◆',
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

def rm_words(word_list):
    new_words = []
    for word in word_list:
        if not TextDeal.isValidWord(word):
            continue
        banned_word = [' ','，',',','.']
        if word in banned_word:
            continue
        new_words.append(word)
    return new_words

def pick_valid_word(word_info_list, dict_size):
    word_info_list.sort(key=lambda x:x['count'],reverse=True)
    word_info_list = word_info_list[:dict_size]
    word2id = {}
    id2word = {}
    for i,line in enumerate(word_info_list):
        word = line['word']
        word2id[word] = i
        id2word[i] = word
    return word2id,id2word

def pick_valid_word_chisquare(word_info_list, dict_size):
    label_list = ['娱乐', '股票', '体育', '科技', '房产', '社会', '游戏', '财经', '时政', '家居', '彩票', '教育', '时尚', '星座']
    word_info_list.sort(key=lambda x:x['count'],reverse=True)
    word_info_list = word_info_list[:50000]
    word2id = {}
    id2word = {}
    for i,line in enumerate(word_info_list):
        word = line['word']
        word2id[word] = i
        id2word[i] = word
    c = ChiSquareCalculator(50000,len(label_list))
    for info in word_info_list:
        word = info['word']
        word_id = word2id[word]
        for item in info['sub_count']:
            c.add(word_id,label_list.index(item),info['sub_count'][item])
    c.cal()
    chi_order = np.argsort(-c.chi_value)[:dict_size]
    valid_word2id = {}
    valid_id2word = {}
    for id in chi_order:
        word = id2word[id]
        valid_word2id[word] = id
        valid_id2word[id] =  word
    return valid_word2id,valid_id2word