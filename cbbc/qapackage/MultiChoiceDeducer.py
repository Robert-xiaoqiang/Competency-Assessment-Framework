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
from .TestHelper import MCResult, compute_accuracy_logits
from .Deducer import Deducer

def to_list(tensor):
    return tensor.detach().cpu().tolist()

class MultiChoiceDeducer(Deducer):
    def __init__(self, model, tokenizer, dev_examples, dev_features, dev_dataloader, config):
        super().__init__(model, tokenizer, dev_examples, dev_features, dev_dataloader, config)

    def build_data(self, batch_data):
        batch_data = [ d.to(self.main_device, non_blocking=True) if torch.is_tensor(d) else d for d in batch_data ]
        inputs = {
            "input_ids": batch_data[0],
            "attention_mask": batch_data[1],
            "token_type_ids": batch_data[2],
            "return_dict": False
        }

        feature_indices = batch_data[3]

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

                # Some models (XLNet, XLM) use 4 arguments for their predictions, while the other "simpler"
                # models only use two.
                if len(outputs_list) >= 4:
                    # TODO
                    pass
                else:
                    choices_logits = outputs_list[0]
                    result = MCResult(unique_id, choices_logits)

                all_results.append(result)

        prediction_dirname = os.path.join(self.prediction_path, 'model_{}'.format(self.config.DEV.SNAPSHOT_KEY))
        os.makedirs(prediction_dirname, exist_ok = True)
        prediction_file = os.path.join(prediction_dirname, 'prediction.json')
        
        self.logger.info("finish feedforwading, let's transform them into option!" )
        if self.config.MODEL.LKEY in ["xlnet", "xlm"]:
            pass
        else:
            predictions = compute_accuracy_logits(
                self.dev_examples,
                self.dev_features,
                all_results,
                prediction_file
            )
        self.logger.info('finish transforming, enjoy yourself!')
 