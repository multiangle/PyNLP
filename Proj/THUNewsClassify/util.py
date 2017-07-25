
import TextDeal
from Proj.THUNewsClassify.chi_square import ChiSquareCalculator
import numpy as np
import collections,random

def rm_chars(sent):
    # 功能: 去除 □ ■ 等无意义的字符
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
    # 功能: 从文件中读取内容，将每行去除无意义字符以后，依次存入列表中
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
    # 功能: 输入单词序列，去掉单词中的 invalid word, 标点符号
    new_words = []
    for word in word_list:
        if not TextDeal.isValidWord(word):
            continue
        banned_word = [' ','，',',','.']
        if word in banned_word:
            continue
        new_words.append(word)
    return new_words

def rm_invalid_words(word_list):
    return rm_words(word_list)

def rm_stop_words(word_list):
    # 功能： 去除单词中的 stop word, 以及标点符号
    new_words = []
    for word in word_list:
        if TextDeal.isStopWord(word):
            continue
        banned_word = [' ','，',',','.']
        if word in banned_word:
            continue
        new_words.append(word)
    return new_words

def pick_valid_word(word_info_list, dict_size):
    # 从输入的词频中挑选出词频前N多的单词
    word_info_list.sort(key=lambda x:x['count'],reverse=True)
    word_info_list = word_info_list[:dict_size]
    word2id = {}
    id2word = {}
    for i,line in enumerate(word_info_list):
        word = line['word']
        word2id[word] = i
        id2word[i] = word
    return word2id,id2word

def pick_valid_word_chisquare(word_info_list, dict_size, s1_size = 50000):
    # 根据单词的卡方值来挑选出合适的单词
    label_list = ['娱乐', '股票', '体育', '科技', '房产', '社会', '游戏', '财经', '时政', '家居', '彩票', '教育', '时尚', '星座']
    # 首先根据词频挑选出前5W个词(其实是为了防止id混乱)
    word_info_list = sorted(word_info_list, key=lambda x:x['count'],reverse=True)
    word_info_list = word_info_list[:s1_size]
    word2id = {}
    id2word = {}
    for i,line in enumerate(word_info_list):
        word = line['word']
        word2id[word] = i
        id2word[i] = word
    c = ChiSquareCalculator(s1_size,len(label_list))
    for info in word_info_list:
        # 依次将单词信息输入chi square矩阵
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

def pick_valid_word_chisquare_concat(word_info_list, dict_size, s1_size=50000):
    # 根据单词的卡方值来挑选出合适的单词, 与非concat的相比，
    # 不再将新的word2id的id范围限制在【0-s1_size】,而是限制在【0-dict_size】
    # WARNING: 该方法不能用于需要预先训练好的词向量的方法，例如MLP, tf中的所有模型

    label_list = ['娱乐', '股票', '体育', '科技', '房产', '社会', '游戏', '财经', '时政', '家居', '彩票', '教育', '时尚', '星座']
    word_info_list = sorted(word_info_list, key=lambda x:x['count'],reverse=True)
    word_info_list_cut = word_info_list[:s1_size]
    word2id={}
    id2word = {}
    c = ChiSquareCalculator(s1_size, len(label_list))
    for i,info in enumerate(word_info_list_cut):
        word = info['word']
        word_id = i
        word2id[word] = i
        id2word[i] = word
        for item in info['sub_count']:
            c.add(word_id, label_list.index(item), info['sub_count'][item])
    c.cal()
    chi_order = np.argsort(-c.chi_value)[:dict_size]
    valid_word2id = {}
    valid_id2word = {}
    for id in chi_order:
        word = id2word[id]
        new_id = len(valid_id2word)
        valid_id2word[new_id] = word
        valid_word2id[word] = new_id
    return valid_word2id, valid_id2word




def gen_balance_samples(file_info_list,label_list,balance_index=2):
    labels = [x['label'] for x in file_info_list]
    label_count = collections.Counter(labels)
    print(label_count)
    freqs = label_count.values()
    min_freq = min(freqs)
    sample_list = [collections.deque(maxlen=min_freq*balance_index) for _ in range(len(label_list))]
    for info in file_info_list:
        labelid = label_list.index(info['label'])
        sample_list[labelid].append(info)
    sample_list = [list(x) for x in sample_list]
    ret = sum(sample_list,[])
    random.shuffle(ret)
    return ret

def gen_balance_samples_withid(file_info_list,label_list,balance_index=2):
    # balance_index 表示样本最多的类和样本最少的类之间的比例
    labels = [x['label'] for x in file_info_list]
    label_count = collections.Counter(labels)
    print(label_count)
    freqs = label_count.values()
    min_freq = min(freqs)
    sample_list = [collections.deque(maxlen=min_freq*balance_index) for _ in range(len(label_list))]
    for i,info in enumerate(file_info_list):
        info['id'] = i
        labelid = label_list.index(info['label'])
        sample_list[labelid].append(info)
    sample_list = [list(x) for x in sample_list]
    ret = sum(sample_list,[])
    random.shuffle(ret)
    return ret