import sys
import os
sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))
import argparse

from qapackage import Architecture
from qapackage.DataProcessor import DataProcessor
from qapackage.Deducer import Deducer
from qapackage.MultiChoiceDeducer import MultiChoiceDeducer

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
    # get instance
    model, tokenizer = Architecture.get_model(config)

    # just for dev stuff, it is unnecessary to use dyanmic DataProcessor
    processor = DataProcessor(config)
    dev_examples, dev_features, dev_dataloader = processor.get_dev_stuff(tokenizer)
    # class
    D = eval(config.DEV.DEDUCER)
    # instantiate
    d = D(model, tokenizer, dev_examples, dev_features, dev_dataloader, config)
    d.deduce()

if __name__ == '__main__':
    main()
