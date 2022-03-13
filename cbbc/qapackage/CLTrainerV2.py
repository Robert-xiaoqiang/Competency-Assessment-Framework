import os
import random
random.seed(32767)
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
from .Trainer import Trainer

'''
    CLTrainerV2 will use itraration based modal suspend/resume, which
    is a main difference from original Trainer
    and use a dynamic sampling strategy, which is different from CLTrainner
'''
class CLTrainerV2(Trainer):
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

        self.num_train_batch_per_epoch = len(self.train_dataloader) # it is incompatible with CL setting, we use it just for fair comparison
        self.num_epochs = self.config.TRAIN.NUM_EPOCHS # as above
        self.num_iterations = self.num_epochs * self.num_train_batch_per_epoch

        ##################################################################
        self.N = len(self.train_features)
        self.full_data_index = list(range(self.N))
        ##################################################################
        self.cur_percent = self.config.CURRICULUM.START_PERCENT

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
        self.loaded_iteration = None
        self.best_result = None

    def build_train_model(self):
        b = False
        if self.config.TRAIN.RESUME:
            b = self.load_checkpoint(snapshot_key = 'latest')
            self.logger.info('loaded successfully, continue training from iteration {}'.format(self.loaded_iteration) \
            if b else 'loaded failed, train from the devil')
        if not b:
            self.logger.info('loaded nothing new')

    def load_checkpoint(self, snapshot_key = 'latest'):
        '''
            load checkpoint and
            make self.loaded_iteration
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
            self.loaded_iteration = params['iteration'] # how to use ????
            return True

    def sample_batch_index(self, batch_index):
        if batch_index and not batch_index % self.config.CURRICULUM.INCREASE_INTERVAL:
            self.cur_percent = min(self.cur_percent * self.config.CURRICULUM.INCREASE_FACTOR, 1)

        limit = np.int(np.ceil(self.N * self.cur_percent))
        cur_data_index = self.full_data_index[:limit]
        target_batch_index = np.random.choice(cur_data_index, self.config.TRAIN.BATCH_SIZE, replace = False)

        return target_batch_index

    def train_epoch(self):
        self.model.train()
        start_iteration = self.loaded_iteration or 0
        end_iteration = self.num_iterations
        for batch_index in range(start_iteration, end_iteration):
            batch_data_index = self.sample_batch_index(batch_index)
            batch_data = self.train_dataset[batch_data_index]
            inputs = self.build_data(batch_data)

            losses, start_logits, end_logits, *leftover = self.model(**inputs)

            # here loss is gathered from each rank, mean/sum it to scalar
            if self.config.TRAIN.REDUCTION == 'mean':
                loss = losses.mean()
            else:
                loss = losses.sum()
            self.on_batch_end(loss, batch_index)

    def on_batch_end(self, loss, batch_index):    
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # lr_scheduler of bert family is called every iteration (not every epoch)
        self.lr_scheduler.step()
        iteration = batch_index + 1

        self.loss_avg_meter.update(loss.item())

        if not iteration % self.config.TRAIN.LOSS_FREQ:
            self.summary_loss(loss, iteration)
        
        if not iteration % self.config.TRAIN.TB_FREQ:
            self.summary_tb(loss, iteration)

        if not iteration % self.config.TRAIN.DEV_FREQ:
            self.on_dev_stage(iteration)

    def on_dev_stage(self, iteration):
        em, f1 = self.validate(iteration)
        self.save_checkpoint(iteration)
        if self.best_result is None or f1 > self.best_result:
            self.save_checkpoint(iteration, 'best')
            self.best_result = f1
        self.writer.add_scalar('val/em', em, iteration)
        self.writer.add_scalar('val/f1', f1, iteration)
        self.model.train()

    def summary_loss(self, loss, iteration):
        self.logger.info('[only one epoch - iteration {}/{}]: loss(cur): {:.4f}, loss(avg): {:.4f}, lr: {:.8f}'\
                .format(iteration, self.num_iterations, loss.item(), self.loss_avg_meter.average(), self.optimizer.param_groups[0]['lr']))

    def save_checkpoint(self, iteration, snapshot_key = 'latest'):
        self.summary_model(iteration, snapshot_key)
        # self.summary_model(iteration, 'latest')

    def summary_model(self, iteration, snapshot_key = 'latest'):
        model_dir_name = os.path.join(self.snapshot_path, 'model_{}'.format(snapshot_key))
        os.makedirs(model_dir_name, exist_ok = True)
        optimizer_lr_state_dict_filename = os.path.join(model_dir_name, 'optimizer_lr_state_dict.ckpt')
        
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(model_dir_name)

        self.tokenizer.save_pretrained(model_dir_name)

        torch.save({ 'optimizer_state_dict': self.optimizer.state_dict(),
                     'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
                     'iteration': iteration
                    }, optimizer_lr_state_dict_filename)

        self.logger.info('save model in {}'.format(model_dir_name))

    def on_train_end(self):
        self.save_checkpoint(self.num_iterations)
        self.logger.info('Finish training with iteration {}, close all'.format(self.num_iterations))
        self.writer.close()

    def train(self):
        self.build_train_model()
        
        self.train_epoch()

        self.on_train_end()