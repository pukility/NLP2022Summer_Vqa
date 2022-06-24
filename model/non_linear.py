import mindspore
import mindspore.nn as nn
# import torch.utils.model_zoo as model_zoo
from mindspore.ops import operations as ops

class non_linear(nn.Cell):
    def __init__(self, cfg):
        super().__init__()
        in_dim = cfg["input_dim"]
        out_dim = cfg["output_dim"]
        self.w1 = nn.Dense(in_channels=in_dim, out_channels=out_dim, has_bias=True)
        self.w2 = nn.Dense(in_channels=in_dim, out_channels=out_dim, has_bias=True)

    def construct(self, x):
        y1 = ops.Tanh()(self.w1(x))
        y2 = ops.Sigmoid()(self.w2(x))
        y = y1 * y2
        return y