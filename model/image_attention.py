import mindspore
import mindspore.nn as nn
# import torch.utils.model_zoo as model_zoo
from mindspore.ops import operations as ops

from .non_linear import non_linear

class image_attention(nn.Cell):
    """image feature extractor.\\ 
    The feature extactor takes a (N, Dq) tensor and a (N, K, Di) tensor as inputs, \\
    with each dim corresponding to batch_size, question feature dimension, imgae regions, image feature dimension
    and output a (N, K, 1) tensor.

    Args:
        cfg (dict): a dict containing the configs
    
    """
    def __init__(self, cfg):
        super().__init__()
        w2_input_size = cfg["image_attention"]["w2_input_size"]
        w2_output_size = cfg["image_attention"]["w2_output_size"]

        self.w1 = non_linear(cfg["w1"])
        self.w2 = nn.Dense(in_channels=w2_input_size, out_channels=w2_output_size)

    def construct(self, img_feat, que_emb):
        # concatenate the image feature and the questio embedding
        K = img_feat.shape[0]
        que_emb = ops.ExpandDims()(que_emb, 1)
        que_emb = ops.Tile()(que_emb, (1, K, 1))
        # feat: (N, K, Dq + Di)
        feat = ops.Concat(1)([img_feat, que_emb])

        attn = self.w1(feat)
        # attn: (N, K, 1)
        attn = self.w2(attn)
        # 这里看看需不需要reshape
        attn = ops.Softmax(1)(attn)
        return attn