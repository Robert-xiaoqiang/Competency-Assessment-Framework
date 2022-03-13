import os
from pprint import pprint

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForQuestionAnswering,
    AlbertTokenizer,
    BertConfig,
    BertForQuestionAnswering,
    BertForMultipleChoice,
    BertTokenizer,
    CamembertConfig,
    CamembertForQuestionAnswering,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForQuestionAnswering,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForQuestionAnswering,
    RobertaTokenizer,
    XLMConfig,
    XLMForQuestionAnswering,
    XLMTokenizer,
    XLNetConfig,
    XLNetForQuestionAnswering,
    XLNetTokenizer,
    get_linear_schedule_with_warmup
)

from .TrainHelper import AverageMeter, LoggerPather


'''
    map from model.lkey to (config, module, tokenizer)
'''
MODELS_MAP = {
    "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
    "camembert": (CamembertConfig, CamembertForQuestionAnswering, CamembertTokenizer),
    "roberta": (RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer),
    "xlnet": (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer),

    "bert-multi-choice": (BertConfig, BertForMultipleChoice, BertTokenizer)
}

def get_model(config):
    CC, MC, TC = MODELS_MAP[config.MODEL.LKEY]
    loggerpather = LoggerPather(config, mode = 'slient')
    snapshot_dir = os.path.join(loggerpather.get_snapshot_path(), 'model_latest')
    sub_key_or_snapshot_dir = config.MODEL.SUBKEY
    # empty or not / ture, false or not ????
    if config.TRAIN.RESUME and os.path.isdir(snapshot_dir) and os.listdir(snapshot_dir):
        sub_key_or_snapshot_dir = snapshot_dir
    pprint('sub_key_or_snapshot_dir is {}'.format(sub_key_or_snapshot_dir))
    model_config = CC.from_pretrained(sub_key_or_snapshot_dir)
    tokenizer = TC.from_pretrained(
        sub_key_or_snapshot_dir,
        do_lower_case = config.TRAIN.DO_LOWER_CASE
    )
    model = MC.from_pretrained(
        sub_key_or_snapshot_dir,
        config = model_config
    )

    return model, tokenizer