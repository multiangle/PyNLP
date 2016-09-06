
import logging
import os
import re

class segmentor_base():
    def __init__(self,
                 stop_words         = None,
                 stop_words_path    = None,
                 filter_words       = True,
                 ):
        # 类内变量： stop_words      - list      - 停用词表
        #           filter_words    - boolean   - 是否过滤词，包括停用词，出现频率过高/过低的词等等

        self.filter_words = filter_words

        # 处理 stop words 和 stop words path
        if stop_words and stop_words_path:          # 同时存在word 和 path 的情况
            logging.warning("stop_words and stop_words_path both exists, use the stop_words")
            self.stop_words = stop_words
        elif stop_words and not stop_words_path:    # 有word 而无 path 的情况
            self.stop_words = stop_words
        elif not stop_words and stop_words_path:    # 没有word 而有 path 的情况
            self.stop_words = self._get_stop_words(stop_words_path)
        elif not stop_words and not stop_words_path:# 两者都没有的情况, 采用默认的路径
            path_project_root = os.getcwd()
            path_stop_words   = os.path.join(path_project_root, 'static','stop_words.txt')
            self.stop_words = self._get_stop_words(path_stop_words)

    def cut(self, content_str):
        """
        cut 函数 ： 调用分词的接口，里面内涵了预处理，分词，以及过滤单词的流程
                    其内部调用了_cut_inner, _filter, _predeal 等函数，可以选择性重载
        :param content_str: 待分词的内容，字符串格式，例如 “我在东北玩泥巴”
        :return: list     : 分好词以后的结果，例如 ["我","在","东北","玩","泥巴"]
        """
        paragraph_list = content_str.split('\n')    # 先按照\n 分隔
        paragraph_list = map(self._predeal, paragraph_list)  #  先进行一下预处理，比如说去掉\u3000等字符
        if self.filter_words:
            words_2d_list = [self._filter(self._cut_inner(para)) for para in paragraph_list]
        else:
            words_2d_list = [self._cut_inner(para) for para in paragraph_list]
        words_list = sum(words_2d_list, [])
        return words_list

    # Override
    def _cut_inner(self, content_str):
        """
        _cut_inner 函数: 是分词的主要函数, 需要被重载的
        :param content_str: 待分词的内容，字符串格式，例如 “我在东北玩泥巴”
        :return: list     : 分好词以后的结果，例如 ["我","在","东北","玩","泥巴"]
        """
        raise RuntimeError('The cut method should be overrided!')
        # return jieba.cut(content_str, cut_all=self.cut_all)

    def _filter(self, word_list):
        """
        _filter : 过滤掉无用词，可以选择重载
        :param word_list:
        :return:
        """
        if type(word_list)!=list:
            word_list = list(word_list)
        for i in range(word_list.__len__())[::-1]:
            if word_list[i] in self.stop_words:
                word_list.pop(i)
            elif word_list[i].isdigit():
                word_list.pop(i)
        return word_list


    def _predeal(self, content_str):
        content_str = re.sub('\u3000','',content_str)
        return content_str

    def _get_stop_words(self, path):
        """
        _get_stop_words: 当没有指定停用词的时候，即采用自带的停用词表
        :param  path    : 存放停用词表的路径，要求每行存放一个词，不可有多余的空格等字符
        :return: list   : 停用词列表，例如['顿时','首先']
        """
        file = open(path,'rb').read().decode('utf8').split('\n')
        file = list(set(file))
        return file


