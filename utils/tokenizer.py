import os
import json
import pickle
import numpy as np

from tqdm import tqdm
from itertools import chain
from os import path as osp
from mindspore import Tensor
from mindspore import dtype as mstype

from .vqaevaluate import VQAEval


splits = ['train', "val", 'test']

class Tokenizer:
    """
    按照下面的流程解析原始数据集，获得features与labels：
    sentence->tokenized->encoded->padding->features
    """

    def __init__(self, cfg):
        que_path = cfg["que_path"]
        ans_path = cfg["ans_path"]
        img_path = cfg["img_path"]
        glove_path = cfg["glove_path"]
        embd_path = cfg["embd_path"]
        embed_size = cfg["embedding"]["embed_size"]

        self.__que_path = que_path
        self.__ans_path = ans_path
        self.__img_path = img_path
        self.__img_file = pickle.load(open(self.__img_path, "rb"))
        self.__embd_path = embd_path
        self.__glove_dim = embed_size
        self.__glove_path = os.path.join(glove_path, 'glove.6B.' + str(self.__glove_dim) + 'd.txt')
        self.__glove_file = open(self.__glove_path, "r")
        self.__glove = {}

        self.__cfg = cfg

        self.__que_text = {}
        self.__que_token = {}
        self.__ans_token = {}
        self.__img_feat = {}

        self.__word2idx = {}
        self.__weight_np = None

    def parse(self):
        """
        解析vqa data
        """
        if osp.exists(osp.join(self.__embd_path, "weight.txt")):
            print("weight data already exists")
        else:
            print("=======================Start parse glove=======================")
            self.__parse_glove()
        


        self.__ans_eval = VQAEval(self.__ans_path, n=8)
        self.__ans_eval.run()
        for split in splits:
            print("=======================Start parse {}=======================".format(split))
            #从原始文本中加载数据
            self.__parse_que_datas(split)
            self.__parse_ans_datas(split)
            #分别读取出文本与label，其中文本处理包括：提取词汇表，文本转词汇id，文本id向量统一长度
            self.__updata_que_to_tokenized(split)
            #生成对应的glove矩阵
            
        self.__construct_dict()
        
        for split in splits:
            #词汇转化为对应id
            self.__encode_features(split)
            #将所有id组成的句子force到同样长度
            self.__padding_features(split, maxlen = self.__cfg["maxlen"])
        
        if not osp.exists(osp.join(self.__embd_path, "weight.txt")):
            self.__gen_weight_np()
            if self.__weight_np is not None:
                np.savetxt(os.path.join(self.__embd_path, 'weight.txt'), self.__weight_np)

    def __parse_glove(self):
        f_lines = self.__glove_file.readlines()

        for line in tqdm(f_lines):
            line = line.split(" ")
            self.__glove[line[0]] = [float(val) for val in line[1:]]

    def __parse_que_datas(self, split):
        """
        加载问题数据，保存为{que_id: que_str}的形式
        """
        path = osp.join(self.__que_path, split+".json")
        que_file = json.load(open(path, "r"))

        que_text = {}
        img_feat = {}
        for q in que_file["questions"]:
            qid = q["question_id"]
            que_text[qid] = q["question"]
            iid = qid//1000
            img_feat[qid] = Tensor(self.__img_file[str(iid)], mstype.float32)

        self.__que_text[split] = que_text
        self.__img_feat[split] = img_feat

    def __parse_ans_datas(self, split):
        """
        加载答案数据，保存为{ans: que_str}的形式
        """

        ans_token = self.__ans_eval.get_acc(split)
        self.__ans_token[split] = ans_token



    def __updata_que_to_tokenized(self, split):
        """
        切分原始语句
        """
        for qid in self.__que_text[split].keys():
            sentence = self.__que_text[split][qid]
            self.__que_text[split][qid] = [word.lower() for word in sentence.split(" ")]


    def __construct_dict(self):
        """
        构建词汇表
        """
        vocab = []
        for s in splits:
            vocab += list(chain(*self.__que_text[s].values()))
        
        vocab = set(vocab)

        # word_to_idx: {'hello': 1, 'world':111, ... '<unk>': 0}
        word_to_idx = {word: i + 1 for i, word in enumerate(vocab)}
        word_to_idx['<unk>'] = 0
        self.__word2idx = word_to_idx


    def __encode_features(self, split):
        """ 
        词汇转化为对应id 
        """
        word_to_idx = self.__word2idx
        encoded_que = {}
        for qid in self.__que_text[split].keys():
            question = self.__que_text[split][qid]
            encoded_sentence = [word_to_idx.get(word, 0) for word in question]
            encoded_que[qid] = encoded_sentence
        self.__que_token[split] = encoded_que


    def __padding_features(self, split, maxlen=14, pad=0):
        """
        将所有id组成的句子force到同样长度
        """
        for qid in self.__que_token[split].keys():
            que_token = self.__que_token[split][qid]
            if len(que_token) >= maxlen:
                padded_que = que_token[:maxlen]
            else:
                padded_que = que_token
                while len(padded_que) < maxlen:
                    padded_que.append(pad)
            self.__que_token[split][qid] = Tensor(padded_que, mstype.int32)


    def __gen_weight_np(self):
        """
        使用gensim获取权重
        """
        weight_np = np.zeros((len(self.__word2idx), self.__glove_dim), dtype=np.float32)
        for word, idx in self.__word2idx.items():
            if word not in self.__glove.keys():
                continue
            word_vector = np.array(self.__glove[word])
            weight_np[idx, :] = word_vector

        self.__weight_np = weight_np


    def get_datas(self, split):
        """
        返回 features, labels, weight
        """
        img = self.__img_feat[split]
        que = self.__que_token[split]
        ans = self.__ans_token[split]
        
        return img, que, ans