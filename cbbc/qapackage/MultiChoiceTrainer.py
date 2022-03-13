import os

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.optim import Adam, SGD, lr_scheduler
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForQuestionAnswering,
    AlbertTokenizer,
    BertConfig,
    BertForQuestionAnswering,
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

from .TrainHelper import AverageMeter, LoggerPather, DeviceWrapper
from .Trainer import Trainer

class MultiChoiceTrainer(Trainer):
    def __init__(self, model, tokenizer, train_dataloader,
                 dev_examples, dev_features, dev_dataloader, config):
        super().__init__(model, tokenizer, train_dataloader,
                         dev_examples, dev_features, dev_dataloader, config)
    
    def build_data(self, batch_data):
        batch_data = [ d.to(self.main_device, non_blocking=True) if torch.is_tensor(d) else d for d in batch_data ]
        inputs = {
            "input_ids": batch_data[0],
            "attention_mask": batch_data[1],
            "token_type_ids": batch_data[2],
            "labels": batch_data[3],
            "return_dict": False
        }

        return inputs

    def train_epoch(self, epoch):
        # set evaluation mode in self.on_epoch_end(), here reset training mode
        self.model.train()
        for batch_index, batch_data in enumerate(self.train_dataloader):
            inputs = self.build_data(batch_data)

            losses, choices_logits, *leftover = self.model(**inputs)
            # here loss is gathered from each rank, mean/sum it to scalar
            if self.config.TRAIN.REDUCTION == 'mean':
                loss = losses.mean()
            else:
                loss = losses.sum()
            self.on_batch_end(loss, epoch, batch_index)