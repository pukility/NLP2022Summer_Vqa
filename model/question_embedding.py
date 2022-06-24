import mindspore
import mindspore.nn as nn
# import torch.utils.model_zoo as model_zoo
from mindspore.ops import operations as ops

model_url = ""

class embedding(nn.Cell):
    """embedding layer.\\ 
    The feature extactor takes a (N, L, V) tensor as input, \\
    with each dim corresponding to batch_size, questio nlength, vocabulary size
    and output a (N, L, D) tensor.
    """
    def __init__(self, cfg):
        super().__init__()
        self._vocab_size = cfg["embedding"]["vocab_size"]
        self._embed_size = cfg["embedding"]["embed_size"]
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
    #     self.load_pretrained_weight()

    # def load_pretrained_weight(self):
        # pretrain_dict = model_zoo.load_url(model_url)
        # self.embedding.load_state_dict(pretrain_dict, strict=False)

    def construct(self, inputs):
        embeds = self.embedding(inputs)
        return embeds

class gru(nn.Cell):
    def __init__(self, cfg):
        super().__init__()
        self._input_size = cfg["gru"]["input_size"]
        self._hidden_size = cfg["gru"]["hidden_sizes"]
        ## 这里num_layer是怎么说
        self.gru = nn.GRU(self._input_size, self._hidden_size, num_layers=1)
        
    def construct(self, inputs):
        ## 这里h0初始化怎么选？
        h0 = ops.ExpandDims()(ops.ZerosLike()(inputs), 0)
        output, hn = self.gru(inputs, h0)
        return output

class question_embedding(nn.Cell):
    """question embedding layer.\\ 
    The feature extactor takes a (N, L, V) tensor as input, \\
    with each dim corresponding to batch_size, questio nlength, vocabulary size
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