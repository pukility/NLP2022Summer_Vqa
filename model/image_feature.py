import mindspore
import mindspore.nn as nn
# import torch.utils.model_zoo as model_zoo
from mindspore.ops import operations as ops

class image_feature(nn.Cell):
    """image feature extractor.\\ 
    The feature extactor takes a (N, H, W, C) tensor as input, \\
    with each dim corresponding to batch_size, height, width, channels
    and output a (N, M, D) tensor.

    Args:
        cfg (dict): a dict containing the configs
    
    """
    def __init__(self, cfg):
        super().__init__()
        self.net_name = cfg["image_feature"]["net_name"]
        
        self.fnet = torch.hub.load("pytorch/vision", self.net_name)
        self.K = cfg["image_feature"]["K"]

    def construct(self, inputs):
        outputs = self.fnet(inputs)
        return outputs
