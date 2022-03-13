import os
import random
random.seed(32767)
import numpy as np
np.random.seed(32767)

import shutil
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
from .TestHelper import QAResult, compute_predictions_logits

class Trainer:
    def __init__(self, model, tokenizer, train_examples, train_features, train_dataset, train_dataloader,
                 dev_examples, dev_features, dev_dataloader, config):
        self.tokenizer = tokenizer # CPU only
        self.model = model
        self.config = config

        cudnn.benchmark = self.config.CUDNN.BENCHMARK
        cudnn.deterministic = self.config.CUDNN.DETERMINISTIC
        cudnn.enabled = self.config.CUDNN.ENABLED

        self.train_examples = train_examples
        self.train_features = train_features
        self.train_dataset = train_dataset
        self.train_dataloader = train_dataloader
        self.num_train_batch_per_epoch = len(self.train_dataloader)
        self.num_epochs = self.config.TRAIN.NUM_EPOCHS
        self.num_iterations = self.num_epochs * self.num_train_batch_per_epoch

        self.dev_examples = dev_examples
        self.dev_features = dev_features
        self.dev_dataloader = dev_dataloader

        self.optimizer = self.build_optimizer() # build self.num_iterations firstly
        self.lr_scheduler = self.build_lr_scheduler()

        self.wrapped_device = DeviceWrapper()(self.config.DEVICE)
        self.main_device = torch.device(self.wrapped_device if self.wrapped_device == 'cpu' else 'cuda:' + str(self.wrapped_device[0]))

        # maybe overwritten in load_checkpoint()
        self.vanilla_model = self.model      
        self.wrap_model()

        loggerpather = LoggerPather(self.config)
        self.logger = loggerpather.get_logger()
        self.snapshot_path = loggerpather.get_snapshot_path()
        self.tb_path = loggerpather.get_tb_path()
        self.prediction_path = loggerpather.get_prediction_path()
        self.writer = SummaryWriter(self.tb_path)
        
        self.loss_avg_meter = AverageMeter()
        
        self.loaded_epoch = None

    def wrap_model(self):
        self.model.to(self.main_device)
        if type(self.wrapped_device) == list:
            self.model = nn.DataParallel(self.model, device_ids = self.wrapped_device)        

    def build_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.TRAIN.WD,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.TRAIN.LR)

        return optimizer

    def build_lr_scheduler(self):
        lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=self.config.TRAIN.NUM_WARMUP_STEPS, num_training_steps=self.num_iterations
        )

        return lr_scheduler

    def build_train_model(self):
        b = False
        if self.config.TRAIN.RESUME:
            b = self.load_checkpoint(snapshot_key = 'latest')
            self.logger.info('loaded successfully, continue training from epoch {}'.format(self.loaded_epoch) \
            if b else 'loaded failed, train from the devil')
        if not b:
            self.logger.info('loaded nothing new')

    def load_checkpoint(self, snapshot_key = 'latest'):
        '''
            load checkpoint and
            make self.loaded_epoch
        '''
        model_dir_name = os.path.join(self.snapshot_path, 'model_{}'.format(snapshot_key))
        if not os.path.isdir(model_dir_name):
            self.logger.info('Cannot find suspended model checkpoint: ' + model_dir_name)
            return False
        else:
            # self.model have been already loaded using .from_pretrained() API
            self.logger.info('Find suspended model checkpoint successfully: ' + model_dir_name)
            map_location = (lambda storage, loc: storage) if self.main_device == 'cpu' else self.main_device
            
            optimizer_lr_state_dict_filename = os.path.join(model_dir_name, 'optimizer_lr_state_dict.ckpt')
            
            params = torch.load(optimizer_lr_state_dict_filename, map_location = map_location)

            self.optimizer.load_state_dict(params['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(params['lr_scheduler_state_dict'])
            self.loaded_epoch = params['epoch']
            return True

    def train_epoch(self, epoch):
        # set evaluation mode in self.on_epoch_end(), here reset training mode
        self.model.train()
        for batch_index, batch_data in enumerate(self.train_dataloader):
            inputs = self.build_data(batch_data)

            losses, start_logits, end_logits, *leftover = self.model(**inputs)

            # here loss is gathered from each rank, mean/sum it to scalar
            if self.config.TRAIN.REDUCTION == 'mean':
                loss = losses.mean()
            else:
                loss = losses.sum()
            self.on_batch_end(loss, epoch, batch_index)

    def build_data(self, batch_data):
        batch_data = [ d.to(self.main_device, non_blocking=True) if torch.is_tensor(d) else d for d in batch_data ]
        inputs = {
            "input_ids": batch_data[0],
            "attention_mask": batch_data[1],
            "token_type_ids": batch_data[2],
            "start_positions": batch_data[3],
            "end_positions": batch_data[4],
            "return_dict": False
        }

        if self.config.MODEL.LKEY in ["xlm", "roberta", "distilbert", "camembert"]:
            del inputs["token_type_ids"]

        if self.config.MODEL.LKEY in ["xlnet", "xlm"]:
            inputs.update({"cls_index": batch_data[5], "p_mask": batch_data[6]})
            if self.config.TRAIN.DATASET_VERSION == 2:
                inputs.update({"is_impossible": batch_data[7]})

        return inputs

    def on_batch_end(self, loss, epoch, batch_index):    
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # lr_scheduler of bert family is called every iteration (not every epoch)
        self.lr_scheduler.step()
        iteration = epoch * self.num_train_batch_per_epoch + batch_index + 1

        self.loss_avg_meter.update(loss.item())

        if not iteration % self.config.TRAIN.LOSS_FREQ:
            self.summary_loss(loss, epoch, iteration)
        
        if not iteration % self.config.TRAIN.TB_FREQ:
            self.summary_tb(loss, iteration)

        if not iteration % self.config.TRAIN.DEV_FREQ:
            em, f1 = self.validate(iteration)
            self.writer.add_scalar('val/em', em, iteration)
            self.writer.add_scalar('val/f1', f1, iteration)
            self.model.train()

    def summary_tb(self, loss, iteration):
        self.writer.add_scalar('train/loss_cur', loss, iteration)
        self.writer.add_scalar('train/loss_avg', self.loss_avg_meter.average(), iteration)

    def summary_loss(self, loss, epoch, iteration):
        self.logger.info('[epoch {}/{} - iteration {}/{}]: loss(cur): {:.4f}, loss(avg): {:.4f}, lr: {:.8f}'\
                .format(epoch, self.num_epochs, iteration, self.num_iterations, loss.item(), self.loss_avg_meter.average(), self.optimizer.param_groups[0]['lr']))

    def on_epoch_end(self, epoch):
        self.save_checkpoint(epoch + 1)
        # val_loss, results = self.validate() # because we have the variable dev_freq, here performing validation is unnecessary, so is the best result updation
        # update_best result
        # self.save_checkpoint(epoch + 1, 'best')
        self.save_checkpoint(epoch + 1, 'epoch_' + str(epoch))

    def save_checkpoint(self, epoch, snapshot_key = 'latest'):
        self.summary_model(epoch, snapshot_key)

    # epoch to resume after suspending or storing
    def summary_model(self, epoch, snapshot_key = 'latest'):
        model_dir_name = os.path.join(self.snapshot_path, 'model_{}'.format(snapshot_key))
        os.makedirs(model_dir_name, exist_ok = True)
        optimizer_lr_state_dict_filename = os.path.join(model_dir_name, 'optimizer_lr_state_dict.ckpt')
        
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(model_dir_name)

        self.tokenizer.save_pretrained(model_dir_name)

        torch.save({ 'optimizer_state_dict': self.optimizer.state_dict(),
                     'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
                     'epoch': epoch
                    }, optimizer_lr_state_dict_filename)

        self.logger.info('save model in {}'.format(model_dir_name))

    def on_train_end(self):
        self.logger.info('Finish training with epoch {}, close all'.format(self.num_epochs))
        self.writer.close()

    def train(self):
        self.build_train_model()
        
        start_epoch = self.loaded_epoch if self.loaded_epoch is not None else 0
        end_epoch = self.num_epochs

        for epoch in range(start_epoch, end_epoch):
            self.train_epoch(epoch)
            self.on_epoch_end(epoch)
        self.on_train_end()

    def build_dev_data(self, batch_data):
        batch_data = [ d.to(self.main_device, non_blocking=True) if torch.is_tensor(d) else d for d in batch_data ]
        inputs = {
            "input_ids": batch_data[0],
            "attention_mask": batch_data[1],
            "token_type_ids": batch_data[2],
            "return_dict": False
        }

        feature_indices = batch_data[3]

        if self.config.MODEL.LKEY in ["xlm", "roberta", "distilbert", "camembert"]:
            del inputs["token_type_ids"]

        if self.config.MODEL.LKEY in ["xlnet", "xlm"]:
            inputs.update({"cls_index": batch_data[5], "p_mask": batch_data[6]})

        return inputs, feature_indices

    def validate(self, iteration):
        self.model.eval()
        all_results = [ ]
        for batch_data in tqdm(self.dev_dataloader, desc = 'Validating'):
            
            inputs, feature_indices = self.build_dev_data(batch_data)
            
            with torch.no_grad():
                outputs = self.model(**inputs)

            for i, feature_index in enumerate(feature_indices):
                eval_feature = self.dev_features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)

                outputs_list = [ o[i].cpu().detach().tolist() for o in outputs ]

                # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
                # models only use two.
                if len(outputs_list) >= 5:
                    start_logits = outputs_list[0]
                    start_top_index = outputs_list[1]
                    end_logits = outputs_list[2]
                    end_top_index = outputs_list[3]
                    cls_logits = outputs_list[4]

                    result = QAResult(
                        unique_id,
                        start_logits,
                        end_logits,
                        start_top_index=start_top_index,
                        end_top_index=end_top_index,
                        cls_logits=cls_logits,
                    )

                else:
                    start_logits, end_logits = outputs_list
                    result = QAResult(unique_id, start_logits, end_logits)

                all_results.append(result)

        prediction_dirname = os.path.join(self.prediction_path, 'model_iteration_{}'.format(iteration))
        os.makedirs(prediction_dirname, exist_ok = True)
        prediction_file = os.path.join(prediction_dirname, 'prediction.json')
        nbest_file = None
        if self.config.MODEL.LKEY in ["xlnet", "xlm"]:
            overall_em_f1_score = None
            pass
        else:
            overall_em_f1_score = compute_predictions_logits(
                self.dev_examples,
                self.dev_features,
                all_results,
                self.config.DEV.N_BEST,
                self.config.DEV.MAX_ANSWER_LENGTH,
                self.config.DEV.DO_LOWER_CASE,
                prediction_file,
                nbest_file,
                False, # verbose_logging
                self.config.TRAIN.DATASET_VERSION,
                self.config.DEV.NULL_SCORE_DIFF_THRESHOLD,
                self.tokenizer,
            )
            backup_dirname = os.path.join(self.prediction_path, 'model_latest')
            os.makedirs(backup_dirname, exist_ok = True)
            # for use of sampling strategy
            shutil.copy(prediction_file, backup_dirname)

        return overall_em_f1_score