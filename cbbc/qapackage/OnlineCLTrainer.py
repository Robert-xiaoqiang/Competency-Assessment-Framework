import os
import json
import random
random.seed(32767)
import shutil

import numpy as np
np.random.seed(32767)
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
from .CLTrainerV2 import CLTrainerV2

class OnlineCLTrainer(CLTrainerV2):
    def __init__(self, model, tokenizer, train_examples, train_features, train_dataset, train_dataloader,
                 dev_examples, dev_features, dev_dataloader, config):
        super().__init__(model, tokenizer, train_examples, train_features, train_dataset, train_dataloader,
                         dev_examples, dev_features, dev_dataloader, config)
        self.cur_percent = None # it is incompatible with online CL setting
        start_limit = np.int(np.ceil(self.N * self.config.CURRICULUM.START_PERCENT))
        self.cur_data_index = self.full_data_index[:start_limit]
        self.cur_data_index_set = set(self.cur_data_index)

    def on_dev_stage(self, iteration):
        em, f1 = self.validate(iteration)
        self.save_checkpoint(iteration)
        # slow training time ????
        factor_threshold = self.get_factor_threshold(iteration)
        if self.best_result is None or f1 > self.best_result:
            self.save_checkpoint(iteration, 'best')
            self.best_result = f1

        self.writer.add_scalar('val/em', em, iteration)
        self.writer.add_scalar('val/f1', f1, iteration)
        for factor_key, factor_score in factor_threshold.items():
            self.writer.add_scalar('val/{}'.format(factor_key.lower()), factor_score, iteration)
    
        self.model.train()

    def get_factor_threshold(self, iteration):
        dev_prediction_dirname = os.path.join(self.prediction_path, 'model_iteration_{}'.format(iteration))
        dev_prediction_file = os.path.join(dev_prediction_dirname, 'prediction.json')
        with open(dev_prediction_file) as f:
            prediction_dict = json.load(f)
        understood_dict = { k: v for k, v in prediction_dict.items() if v['f1_score'] >= 0.8001 and v['em_score'] >= 0.8001 }
 
        threshold_dict = { }
        for factor_entry in self.config.CURRICULUM.DEV_FACTORS:
            factor_key, factor_keyed_difficulty_filename = list(factor_entry.items())[0]
            factor_scores = 0.0
            with open(factor_keyed_difficulty_filename) as f:
                factor_keyed_difficulty_dict = json.load(f)
            for qid in understood_dict.keys():
                factor_scores += factor_keyed_difficulty_dict[qid]
            if understood_dict:
                factor_scores /= float(len(understood_dict))
            threshold_dict[factor_key] = factor_scores

        dev_factor_threshold_file = os.path.join(dev_prediction_dirname, 'factor_threshold.json')
        with open(dev_factor_threshold_file, 'w') as f:
            json.dump(threshold_dict, f, indent = 4)

        backup_dirname = os.path.join(self.prediction_path, 'model_latest')
        os.makedirs(backup_dirname, exist_ok = True)
        # for use of sampling strategy
        shutil.copy(dev_factor_threshold_file, backup_dirname)

        return threshold_dict

    def enlarge_data_index(self):
        latest_factor_threshold_file = os.path.join(self.prediction_path, 'model_latest', 'factor_threshold.json')
        with open(latest_factor_threshold_file) as f:
            threshold_dict = json.load(f)

        candidates = set()
        # for each factor -> filter all samples -> add it into set
        for factor_entry in self.config.CURRICULUM.TRAIN_FACTORS:
            factor_key, factor_keyed_difficulty_filename = list(factor_entry.items())[0]
            factor_threshold_score = threshold_dict[factor_key]
            increased_score = factor_threshold_score * self.config.CURRICULUM.FACTOR_INCREASE_FACTOR
            with open(factor_keyed_difficulty_filename) as f:
                factor_keyed_difficulty_dict = json.load(f)
            for qid, factor_score in factor_keyed_difficulty_dict.items():
                if factor_score < increased_score:
                    # remove duplicates automatically
                    candidates.add(qid)

        new_data_index = [ ]
        new_data_index_set = set()
        # enlarge 2 times
        enlarge_size = len(self.cur_data_index)
        # note that candidates is example-based counting.
        # we cannot use np.sample.choice() because dataset is feature-based counting !!!
        for feature_index, feature in enumerate(self.train_features):
            example = self.train_examples[feature.example_index]
            qid = example.qas_id
            if qid in candidates and feature_index not in self.cur_data_index_set:
                new_data_index.append(feature_index)
                new_data_index_set.add(feature_index)

                if len(new_data_index) == enlarge_size:
                    break

        self.cur_data_index.extend(new_data_index) # +=
        self.cur_data_index_set.update(new_data_index_set) # |=, union_update

    def sample_batch_index(self, batch_index):
        if batch_index and not batch_index % self.config.CURRICULUM.INCREASE_INTERVAL:
            self.enlarge_data_index()
            self.writer.add_scalar('cl/n_data', len(self.cur_data_index_set), batch_index + 1)

        target_batch_index = np.random.choice(self.cur_data_index, self.config.TRAIN.BATCH_SIZE, replace = False)

        return target_batch_index