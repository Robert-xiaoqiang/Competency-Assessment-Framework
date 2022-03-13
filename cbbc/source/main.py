import sys
import os
sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))
import argparse

from qapackage import Architecture

from qapackage.DataProcessor import DataProcessor
from qapackage.CLDataProcessor import CLDataProcessor
from qapackage.CLDataProcessorV2 import CLDataProcessorV2
from qapackage.OnlineCLDataProcessor import OnlineCLDataProcessor

from qapackage.Trainer import Trainer
from qapackage.MultiChoiceTrainer import MultiChoiceTrainer
from qapackage.CLTrainer import CLTrainer
from qapackage.CLTrainerV2 import CLTrainerV2
from qapackage.OnlineCLTrainer import OnlineCLTrainer
from qapackage.SlackOnlineCLTrainer import SlackOnlineCLTrainer

from configure.default import config, update_config

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line interface",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    parse_args()
    # instantiate
    model, tokenizer = Architecture.get_model(config)

    # get class
    D = eval(config.TRAIN.DATAPROCESSOR)
    # instantiate
    processor = D(config)
    train_examples, train_features, train_dataset, train_dataloader = processor.get_train_stuff(tokenizer)
    dev_examples, dev_features, dev_dataloader = processor.get_dev_stuff(tokenizer)
    # get class
    T = eval(config.TRAIN.TRAINER)
    # instantiate
    t = T(model, tokenizer, train_examples, train_features, train_dataset, train_dataloader,
          dev_examples, dev_features, dev_dataloader, config)
    t.train()

if __name__ == '__main__':
    main()
