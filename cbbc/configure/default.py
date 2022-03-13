import os

from yacs.config import CfgNode as CN

_C = CN()

_C.NAME = ''
_C.DEVICE = ''
_C.SEED = 32767
_C.SUMMARY_DIR = ''

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# pretext or surrgate task pretraining
_C.PRETEXT = CN(new_allowed = True)

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.LKEY = ''
_C.MODEL.SUBKEY = ''
_C.MODEL.EXTRA = CN(new_allowed=True)

# contrastive learning
_C.CONTRASTIVE = CN(new_allowed=True)

# curriulum learning
_C.CURRICULUM = CN(new_allowed=True)

# training
_C.TRAIN = CN()
_C.TRAIN.TRAINER = ''
_C.TRAIN.DATAPROCESSOR = ''
_C.TRAIN.PROCESSOR = ''
_C.TRAIN.CONVERTER = ''
_C.TRAIN.DATASET_VERSION = 1
_C.TRAIN.DATASET_FILENAME = ''

_C.TRAIN.MAX_INPUT_LENGTH = 384
_C.TRAIN.DOC_STRIDE = 128
_C.TRAIN.MAX_QUERY_LENGTH = 64

_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.WORKERS = 4
_C.TRAIN.NUM_EPOCHS = 288
_C.TRAIN.NUM_WARMUP_STEPS = 320
_C.TRAIN.RESUME = True
_C.TRAIN.LOSS_FREQ = 10
_C.TRAIN.TB_FREQ = 10
_C.TRAIN.DEV_FREQ = 10
_C.TRAIN.UNLABELED = CN(new_allowed=True)

_C.TRAIN.LR = 0.001
_C.TRAIN.EXTRA_LR = 0.0001
_C.TRAIN.LD = 0.9
_C.TRAIN.WD = 5.0e-4
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.NESTEROV = False
_C.TRAIN.REDUCTION = 'mean'

_C.TRAIN.DO_LOWER_CASE = False

# validating
_C.DEV = CN()
_C.DEV.DEDUCER = ''
_C.DEV.PROCESSOR = ''
_C.DEV.CONVERTER = ''
_C.DEV.DATASET_FILENAME = ''
_C.DEV.MAX_INPUT_LENGTH = 384
_C.DEV.DOC_STRIDE = 128
_C.DEV.MAX_QUERY_LENGTH = 64

_C.DEV.BATCH_SIZE = 8
_C.DEV.WORKERS = 4
_C.DEV.SNAPSHOT_KEY = ''

_C.DEV.N_BEST = 20
_C.DEV.DO_LOWER_CASE = False
_C.DEV.MAX_ANSWER_LENGTH = 50
_C.DEV.NULL_SCORE_DIFF_THRESHOLD = 0.0

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = False
_C.DEBUG.SAVE_HEATMAPS_PRED = False

config = _C

def update_config(cfg, args):
    cfg.defrost()
    
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()

if __name__ == '__main__':
    import sys
    import os
    filename = os.path.join(os.path.dirname(__file__), sys.argv[1])
    with open(filename, 'w') as f:
        print(_C, file=f)
