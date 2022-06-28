import mindspore.nn as nn
from mindspore.ops import operations as ops

from .non_linear import non_linear
from .image_attention import image_attention
from .question_embedding import question_embedding


class vqa_model(nn.Cell):
    def __init__(self, cfg):
        super().__init__()
        self.q_embd = question_embedding(cfg)
        self.i_attn = image_attention(cfg)

        self.i_w1 = non_linear(cfg["i_w1"])
        self.i_w2 = non_linear(cfg["i_w2"])
        self.i_w3 = nn.Dense(in_channels=cfg['i_w3']['input_dim'], out_channels=cfg['i_w3']['output_dim'])

        self.q_w1 = non_linear(cfg["q_w1"])
        self.q_w2 = non_linear(cfg["q_w2"])
        self.q_w3 = nn.Dense(in_channels=cfg['q_w3']['input_dim'], out_channels=cfg['q_w3']['output_dim'])

    def construct(self, img, que):
        q_embd = self.q_embd(que)
        # i_attn: (N, K, 1)
        i_attn = self.i_attn(img, q_embd)
        i_feat = img * i_attn
        # Stacked
        i_attn = self.i_attn(i_feat, q_embd)
        i_feat = img * i_attn
        # i_feat: (N, D)
        i_feat = ops.ReduceSum()(i_feat, 1)

        q_embd = self.q_w1(q_embd)
        i_feat = self.i_w1(i_feat)
        fusion_feat = q_embd * i_feat

        i_feat = self.i_w2(fusion_feat)
        i_feat = self.i_w3(i_feat)
        q_embd = self.q_w2(fusion_feat)
        q_embd = self.q_w3(q_embd)

        res = ops.Sigmoid()(i_feat + q_embd)

        return res