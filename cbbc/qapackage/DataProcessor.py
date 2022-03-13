import json

from torch.utils.data import DataLoader, Sampler, RandomSampler, SequentialSampler
import numpy as np

from .SquadProcessor import SquadProcessor, SquadConverter
from .HotpotProcessor import HotpotProcessor, HotpotConverter
from .RACEProcessor import RACEProcessor, RACEConverter

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.processor, self.converter = eval(self.config.TRAIN.PROCESSOR)(), eval(self.config.TRAIN.CONVERTER)()

    def get_train_stuff(self, tokenizer):
        train_examples = self.processor.get_train_examples(self.config.TRAIN.DATASET_FILENAME)
        train_features, train_dataset = self.converter.convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=self.config.TRAIN.MAX_INPUT_LENGTH,
            doc_stride=self.config.TRAIN.DOC_STRIDE,
            max_query_length=self.config.TRAIN.MAX_QUERY_LENGTH,
            is_training=True,
            return_dataset="pt",
            threads=self.config.TRAIN.WORKERS,
        )
        sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=self.config.TRAIN.BATCH_SIZE)

        return train_examples, train_features, train_dataset, train_dataloader

    def get_dev_stuff(self, tokenizer):
        dev_examples = self.processor.get_dev_examples(self.config.DEV.DATASET_FILENAME)
        dev_features, dev_dataset = self.converter.convert_examples_to_features(
            examples=dev_examples,
            tokenizer=tokenizer,
            max_seq_length=self.config.DEV.MAX_INPUT_LENGTH,
            doc_stride=self.config.DEV.DOC_STRIDE,
            max_query_length=self.config.DEV.MAX_QUERY_LENGTH,
            is_training=False,
            return_dataset="pt",
            threads=self.config.DEV.WORKERS,
        )
        dev_sampler = SequentialSampler(dev_dataset)
        dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=self.config.DEV.BATCH_SIZE)

        return dev_examples, dev_features, dev_dataloader
