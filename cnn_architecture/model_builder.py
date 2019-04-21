import torch.nn.functional as F
from torch import nn
from yacs.config import CfgNode as CN

from blocks import AddBlock, BNBetweenBlock, BNPostBlock, BNPreBlock, SimpleBlock


class ModelTemplate(nn.Module):
    def __init__(self, block_type, conv_cfg: CN, num_classes: int):
        super().__init__()
        self._conv = nn.Conv2d(conv_cfg.IN_CHANNELS, conv_cfg.NUM_FILTERS, conv_cfg.SIZE, stride=conv_cfg.STRIDE,
                               padding=conv_cfg.SIZE // 2)

        self._block1 = block_type(conv_cfg.NUM_FILTERS)
        self._block2 = block_type(self._block1.num_hidden)
        self._cls = nn.Linear(self._block2.num_hidden, num_classes)

    def forward(self, batch):
        batch = F.relu(self._conv(batch))
        batch = self._block1(batch)
        batch = self._block2(batch)
        batch = F.max_pool2d(batch, batch.shape[-2:])
        batch = self._cls(batch.view(batch.shape[:2]))
        return batch


_block_factory = {
    'simple':     SimpleBlock,
    'add':        AddBlock,
    'bn_pre':     BNPreBlock,
    'bn_between': BNBetweenBlock,
    'bn_post':    BNPostBlock
}


def create_model(model_cfg: CN):
    return ModelTemplate(_block_factory[model_cfg.BLOCK_TYPE], model_cfg.CONV0, model_cfg.NUM_CLASSES)
