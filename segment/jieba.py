
import jieba
from segment.segmentor_base import segmentor_base

class segmentor(segmentor_base):
    def __init__(self, cut_all=True, stop_words=None):
        segmentor_base.__init__(self, stop_words=stop_words)
        self.cut_all = cut_all

    #Override
    def _cut_inner(self, content_str):
        return jieba.cut(content_str,cut_all=self.cut_all)

