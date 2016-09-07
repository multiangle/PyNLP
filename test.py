
import numpy as np
from scipy.sparse import csr_matrix
from scipy import sparse
import jieba

content = """北京北京

当我走在这里的每一条街道
我的心似乎从来都不能平静
除了发动机的轰鸣和电气之音
我似乎听到了他烛骨般的心跳
我在这里欢笑
我在这里哭泣
我在这里活着
也在这里死去"""

content = list(jieba.cut(content,cut_all=True))
print(content)
