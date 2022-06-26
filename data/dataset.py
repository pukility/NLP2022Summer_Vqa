import cv2
import os
import numpy as np
import mindspore.dataset as ds
from ..utils.tokenizer import Tokenizer

class MyDataset:
    """自定义数据集类"""

    def __init__(self, cfg, parser = None, split = "train"):
        """自定义初始化操作"""
        self._que = []
        self._ans = []
        self._img = []
        
        if parser == None:
            parser = Tokenizer(cfg)
            parser.parse()
        
        img, que, ans, _ = parser.get_datas(split)
        qids = que.keys()
        for qid in qids:
            self._img.append(img[qid])
            self._que.append(que[qid])
            self._ans.append(ans[qid])
            
    def __getitem__(self, index):
        """自定义随机访问函数"""
        return self._img[index], self._que[index], self._ans[index]

    def __len__(self):
        """自定义获取样本数据量函数"""
        return len(self._que)

def build_dataset(cfg, parser=None, split = "train"):
    batch_size = cfg["batch_size"] 
    dataset = MyDataset(cfg, parser, split)
    dataset = ds.GeneratorDataset(dataset, column_names=["img", "que", "ans"])
    dataset = dataset.batch(batch_size=batch_size)
    return dataset

