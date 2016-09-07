
from gensim import corpora,models
from scipy.sparse import csr_matrix
import numpy as np
import os

class doc2bow():
    """
    这部分实现的是将文本从单词序列转化成按频率表示的向量的过程
    """
    def __init__(self, dictionary):

        assert type(dictionary)==corpora.Dictionary
        self.dictionary = dictionary

    def __getitem__(self, word_list):
        return self.dictionary.doc2bow(word_list)

    def batch_deal(self, doc_list, csr_matrix_output=False):
        """
        能够批量将分好词以后的文档集转化成词频表示的向量集
        :param doc_list:            分好词以后的文档集,是一个二维列表 dim1 代表各文档 dim2代表单个文档中单词
        :param csr_matrix_output:   输出格式，是否要转化成csr_matrix输出
        :param store_folder：       是否要持久化，若是，则指定存储位置
        :param store_file:          如果要存储，可以指定存储的文件名
        :return:
        """
        if not csr_matrix_output:  # 如果不需要做转换
            corpus_bow = [self[doc] for doc in doc_list]





class tf_idf_model():
    def __init__(self, dictionary):
        assert type(dictionary)==corpora.Dictionary
        self.dictionary = dictionary

    def __getitem__(self, bow ):
        # todo
        pass

class lsi_model():
    pass
    # todo

def corpus2csrmatrix(corpus, vec_len=None, divide=None):
    '''
    将gensim中corpus 改成 scipy中的csr_matrix, 不过这个只提供将所有corpus生成以后，统一转成csr matrix
    :param corpus: [(col,value)...]
    :param vec_len: 向量的长度
    :param divide:　如果需要将corpus等比例放入不同的矩阵中,可以指定divide,表示不同矩阵的比例
                    例如要分成8:２的　则divide = [0.8]
                    要分成6:2:2的　则divide = [0.6,0.8],　以此类推
    :return: csr_matrix
    '''
    if not divide:
        data = []
        cols = []
        rows = []
        line_count = 0
        for line in corpus:
            for item in line:
                rows.append(line_count)
                cols.append(item[0])
                data.append(item[1])
            line_count += 1
        if vec_len:
            c_matrix = csr_matrix((data,(rows,cols)),shape=(line_count,vec_len))
        else:
            c_matrix = csr_matrix((data,(rows,cols)))
        return c_matrix
    else:
        assert type(divide)==list   #　确保输入的是list
        assert divide.__len__()>0   #  确保list非空
        latest=0
        for i in divide:
            assert i<1 and i>latest #  确保list内值递增且<1
            latest = i

        batch_num = divide.__len__()+1
        data_list = [[] for i in range(batch_num)]
        cols_list = [[] for i in range(batch_num)]
        rows_list = [[] for i in range(batch_num)]
        count_list = [0] * batch_num
        for line in corpus:
            # 确定该数据属于哪类
            cate = None
            r_value = np.random.random() # 产生随机数
            for i in range(batch_num-1):
                if r_value < divide[i]:
                    cate = i
                    break
            if not cate:
                cate = batch_num-1

            # 将相应数值放入 col, row 和 data
            for item in line:
                rows_list[cate].append(count_list[cate])
                cols_list[cate].append(item[0])
                data_list[cate].append(item[1])
            count_list[cate] += 1

        # 生成 csr_matrix 列表

        matrix_list = []
        if vec_len:
            for i in range(batch_num):
                tmp_matrix = csr_matrix((data_list[i],(rows_list[i],cols_list[i])),shape=(count_list[i],vec_len))
                matrix_list.append(tmp_matrix)
            return matrix_list
        else:
            for i in range(batch_num):
                tmp_matrix = csr_matrix((data_list[i],(rows_list[i],cols_list[i])))
                matrix_list.append(tmp_matrix)
            return matrix_list

class csr_matrix_builder():
    def __init__(self):
        self.data = []
        self.cols = []
        self.rows = []




