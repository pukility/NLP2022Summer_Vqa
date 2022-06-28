import mindspore
import mindspore.dataset as ds
from ..utils.tokenizer import Tokenizer

def build_dataset(cfg, parser=None, split="train"):
    dataset = []

    if parser == None:
        parser = Tokenizer(cfg)
        parser.parse()

    img, que, ans = parser.get_datas(split)
    qids = que.keys()
    for qid in qids:
        dataset.append([img[qid], que[qid], ans[qid]])

    dataset = ds.GeneratorDataset(dataset, column_names=["img", "que", "ans"], column_types=[mindspore.float32, mindspore.int32, mindspore.float32])
    dataset = dataset.batch(batch_size=cfg["batch_size"] , drop_remainder=True)
    return dataset
