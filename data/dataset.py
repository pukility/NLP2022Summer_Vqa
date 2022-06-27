import mindspore
import mindspore.dataset as ds
from ..utils.tokenizer import Tokenizer

# class MyDataset:
#     """自定义数据集类"""

#     def __init__(self, cfg, parser = None, split = "train"):
#         """自定义初始化操作"""
#         self._que = []
#         self._ans = []
#         self._img = []
        
#         if parser == None:
#             parser = Tokenizer(cfg)
#             parser.parse()
        
#         img, que, ans = parser.get_datas(split)
#         qids = que.keys()
#         for qid in qids:
#             self._img.append(img[qid])
#             self._que.append(que[qid])
#             self._ans.append(ans[qid])
#         print(self._img[0])
#         print(self._que[0])
#         print(self._ans[0])
            
#     def __getitem__(self, index):
#         """自定义随机访问函数"""
#         return self._img[index], self._que[index], self._ans[index]

#     def __len__(self):
#         """自定义获取样本数据量函数"""
#         return len(self._que)

def build_dataset(cfg, parser=None, split = "train"):
    datalist = []

    if parser == None:
        parser = Tokenizer(cfg)
        parser.parse()

    img, que, ans = parser.get_datas(split)
    qids = que.keys()
    for qid in qids:
        datalist.append([img[qid], que[qid], ans[qid]])

    batch_size = cfg["batch_size"] 
#     dataset = MyDataset(cfg, parser, split)
    dataset = datalist
    dataset = ds.GeneratorDataset(dataset, column_names=["img", "que", "ans"], column_types=[mindspore.float32, mindspore.int32, mindspore.float32])
    dataset = dataset.batch(batch_size=batch_size)
    return dataset

