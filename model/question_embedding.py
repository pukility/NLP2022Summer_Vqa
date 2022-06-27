import mindspore
import numpy as np
import os.path as osp
import mindspore.nn as nn

from mindspore.ops import operations as ops
from mindspore import Tensor, Parameter

class embedding(nn.Cell):
    """embedding layer.\\ 
    The feature extactor takes a (N, L) tensor as input, \\
    with each dim corresponding to batch_size, question length, vocabulary size
    and output a (N, L, D) tensor.
    """
    def __init__(self, cfg):
        super().__init__()
        self._vocab_size = cfg["embedding"]["vocab_size"]
        self._embed_size = cfg["embedding"]["embed_size"]

        self._embdpath = osp.join(cfg["embd_path"], "weight.txt")
        self._embedding_table = Tensor(np.loadtxt(self._embdpath).astype(np.float32))
        self.embedding = nn.Embedding(self._vocab_size, self._embed_size, embedding_table = self._embedding_table)

    def construct(self, inputs):
        embeds = self.embedding(inputs.astype('int32'))
        return embeds

class gru(nn.Cell):
    def __init__(self, cfg):
        super().__init__()
        self._input_size = cfg["gru"]["input_size"]
        self._hidden_size = cfg["gru"]["hidden_sizes"]
        
        stdv = 1 / np.sqrt(self._hidden_size)
        shape = (1, cfg["batch_size"], self._hidden_size)
        self.gru = nn.LSTM(self._input_size, self._hidden_size, batch_first = True)
        self.h0 = Parameter(Tensor(np.random.uniform(-stdv, stdv, shape).astype(np.float16)))
        self.c0 = Parameter(Tensor(np.random.uniform(-stdv, stdv, shape).astype(np.float16)))
        
        self.trans = ops.Transpose()

    def construct(self, inputs):
        ## 这里h0初始化怎么选？
        output, _ = self.gru(inputs, (self.h0, self.c0))
        output = self.trans(output, (1, 0, 2))[:, -1, :]
        return output

class question_embedding(nn.Cell):
    """question embedding layer.\\ 
    The feature extactor takes a (N, L, V) tensor as input, \\
    with each dim corresponding to batch_size, question length, vocabulary size
    and output a (N, D) tensor.
    """
    def __init__(self, cfg):
        super().__init__()
        self.embd = embedding(cfg)
        self.gru = gru(cfg)

    def construct(self, inputs):
        res = self.embd(inputs)
        res = self.gru(res)
        return res