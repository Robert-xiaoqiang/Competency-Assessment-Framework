import os
import json

from torch.utils.data import DataLoader, Sampler, RandomSampler, SequentialSampler
import numpy as np
from pprint import pprint

from .SquadProcessor import SquadProcessor, SquadConverter
from .HotpotProcessor import HotpotProcessor, HotpotConverter
from .RACEProcessor import RACEProcessor, RACEConverter

from .DataProcessor import DataProcessor
from .TrainHelper import sort_by_order_file

class CLDataProcessorV2(DataProcessor):
    def __init__(self, config):
        super().__init__(config)

    def get_train_stuff(self, tokenizer):
        train_examples = self.processor.get_train_examples(self.config.TRAIN.DATASET_FILENAME)
        if hasattr(self.config.CURRICULUM, 'ORDER_FILENAME') and os.path.isfile(self.config.CURRICULUM.ORDER_FILENAME):
            pprint('Using order filename {}, note that it is a valid behavior only under pre-defined sampling setting, take a double-check to ensure validity'.format(self.config.CURRICULUM.ORDER_FILENAME))

            train_examples = sort_by_order_file(train_examples, self.config.CURRICULUM.ORDER_FILENAME)
            # train_dataset has the same order as train_examples
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
            sampler = SequentialSampler(train_dataset)
            # dataloader in CLTrainerV2 is just for computation of num_iterations
            # data will be fetched from train_data(TensorDataset) diectly
            train_dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=self.config.TRAIN.BATCH_SIZE)
        else:
            raise NotImplementedError('Invalid order filename')

        return train_examples, train_features, train_dataset, train_dataloader