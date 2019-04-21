from pathlib import Path

import numpy as np
import torch
from yacs.config import CfgNode as CN

_cfg = CN()
_cfg.NAME = ''
_cfg.OUTPUT_DIR = ''
_cfg.SEED = 42

_cfg.MODEL = CN()
_cfg.MODEL.CONV0 = CN()
_cfg.MODEL.CONV0.IN_CHANNELS = 3
_cfg.MODEL.CONV0.NUM_FILTERS = 128
_cfg.MODEL.CONV0.SIZE = 7
_cfg.MODEL.CONV0.STRIDE = 2

_cfg.MODEL.NUM_CLASSES = 100
_cfg.MODEL.BLOCK_TYPE = 'simple'

_cfg.TRAIN = CN()
_cfg.TRAIN.LR = 0.1
_cfg.TRAIN.MOMENTUM = 0.9
_cfg.TRAIN.WEIGHT_DECAY = 0.0005
_cfg.TRAIN.NUM_ITERS = 10_000
_cfg.TRAIN.BATCH_SIZE = 256
_cfg.TRAIN.SHUFFLE = True

_cfg.VAL = CN()
_cfg.VAL.BATCH_SIZE = 0


def load_cfg(cfg_path: Path):
    cfg = _cfg.clone()
    cfg.merge_from_file(cfg_path)
    cfg = _transform_cfg(cfg, cfg_path.stem)

    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    _update_seeds(cfg.SEED)
    return cfg


def _transform_cfg(cfg: CN, name: str):
    if cfg.NAME == '':
        cfg.NAME = name
    if cfg.OUTPUT_DIR == '':
        cfg.OUTPUT_DIR = f"output/{cfg.NAME}"
    if cfg.VAL.BATCH_SIZE == 0:
        cfg.VAL.BATCH_SIZE = cfg.TRAIN.BATCH_SIZE * 2
    return cfg


def _update_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
