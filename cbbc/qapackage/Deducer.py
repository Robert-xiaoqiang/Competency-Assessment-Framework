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
from .TestHelper import QAResult, compute_predictions_logits

def to_list(tensor):
    return tensor.detach().cpu().tolist()

class Deducer:
    def __init__(self, model, tokenizer, dev_examples, dev_features, dev_dataloader, config):
        self.tokenizer = tokenizer # CPU only
        self.model = model
        self.config = config

        cudnn.benchmark = self.config.CUDNN.BENCHMARK
        cudnn.deterministic = self.config.CUDNN.DETERMINISTIC
        cudnn.enabled = self.config.CUDNN.ENABLED

        self.dev_examples = dev_examples
        self.dev_features = dev_features
        self.dev_dataloader = dev_dataloader

        self.wrapped_device = DeviceWrapper()(self.config.DEVICE)
        self.main_device = torch.device(self.wrapped_device if self.wrapped_device == 'cpu' else 'cuda:' + str(self.wrapped_device[0]))

        # maybe overwritten in self.load_checkpoint()
        self.vanilla_model = self.model      
        self.wrap_model()

        loggerpather = LoggerPather(self.config)
        self.logger = loggerpather.get_logger()
        self.snapshot_path = loggerpather.get_snapshot_path()
        self.tb_path = loggerpather.get_tb_path()
        self.prediction_path = loggerpather.get_prediction_path()
        self.writer = SummaryWriter(self.tb_path)
        
        self.loaded_epoch = None

    def wrap_model(self):
        self.model.to(self.main_device)
        if type(self.wrapped_device) == list:
            self.model = nn.DataParallel(self.model, device_ids = self.wrapped_device)        

    def build_test_model(self, snapshot_key = None):
        b = self.load_checkpoint(snapshot_key = snapshot_key or self.config.DEV.SNAPSHOT_KEY)
        if b:
            self.logger.info('loaded successfully, test based on model from best epoch {}'.format(self.loaded_epoch))
        else:
            self.logger.info('loaded failed, test based on ImageNet scratch')

    def load_checkpoint(self, snapshot_key):
        '''
            load checkpoint and
            make self.loaded_epoch
        '''
        model_dir_name = os.path.join(self.snapshot_path, 'model_{}'.format(snapshot_key))
        if not os.path.isdir(model_dir_name):
            self.logger.info('Cannot find suspended model checkpoint: ' + model_dir_name)
            return False
        else:
            # during architecture.get_model()
            # self.model have been already loaded using .from_pretrained() API
            self.logger.info('Find suspended model checkpoint successfully: ' + model_dir_name)
            map_location = (lambda storage, loc: storage) if self.main_device == 'cpu' else self.main_device
            
            optimizer_lr_state_dict_filename = os.path.join(model_dir_name, 'optimizer_lr_state_dict.ckpt')
            
            params = torch.load(optimizer_lr_state_dict_filename, map_location = map_location)

            self.loaded_epoch = params.get('epoch') or params.get('iteration')
            return True

    def build_data(self, batch_data):
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

    def deduce(self):
        self.build_test_model()
        self.model.eval()
        all_results = [ ]
        for batch_data in tqdm(self.dev_dataloader, desc = 'Evaluating'):
            inputs, feature_indices = self.build_data(batch_data)
            
            with torch.no_grad():
                outputs = self.model(**inputs)

            for i, feature_index in enumerate(feature_indices):
                eval_feature = self.dev_features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)

                outputs_list = [ to_list(o[i]) for o in outputs ]

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

        prediction_dirname = os.path.join(self.prediction_path, 'model_{}'.format(self.config.DEV.SNAPSHOT_KEY))
        os.makedirs(prediction_dirname, exist_ok = True)
        prediction_file = os.path.join(prediction_dirname, 'prediction.json')
        nbest_file = os.path.join(prediction_dirname, 'nbest.json')
        
        self.logger.info("finish feedforwading, let's transform them into text!" )
        if self.config.MODEL.LKEY in ["xlnet", "xlm"]:
            pass
        else:
            em, f1 = compute_predictions_logits(
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
        self.logger.info('finish transforming with em: {:.4f}, f1: {:.4f}, enjoy yourself!'.format(em, f1))

    def traverse_deduce(self):
        for iteration in range(10, 20000, 10):
            snapshot_key = 'iteration_' + str(iteration)
            self.build_test_model(snapshot_key = snapshot_key)
            self.model.eval()
            all_results = [ ]
            for batch_data in tqdm(self.dev_dataloader, desc = 'Evaluating'):
                inputs, feature_indices = self.build_data(batch_data)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)

                for i, feature_index in enumerate(feature_indices):
                    eval_feature = self.dev_features[feature_index.item()]
                    unique_id = int(eval_feature.unique_id)

                    outputs_list = [ to_list(o[i]) for o in outputs ]

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

            prediction_dirname = os.path.join(self.prediction_path, 'model_{}'.format(snapshot_key))
            os.makedirs(prediction_dirname, exist_ok = True)
            prediction_file = os.path.join(prediction_dirname, 'prediction.json')
            nbest_file = os.path.join(prediction_dirname, 'nbest.json')
            
            self.logger.info("finish feedforwading, let's transform them into text!" )
            if self.config.MODEL.LKEY in ["xlnet", "xlm"]:
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
            self.logger.info('finish transforming, enjoy yourself!')
