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

class CLDataProcessor(DataProcessor):
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
            total_iterations = self.config.TRAIN.NUM_EPOCHS * ((len(train_dataset) + self.config.TRAIN.BATCH_SIZE - 1) // self.config.TRAIN.BATCH_SIZE)
            batch_sampler = CurriculumBatchSampler(
                data_source = train_dataset,
                batch_size = self.config.TRAIN.BATCH_SIZE,
                drop_last = False,
                total_iterations = total_iterations,
                start_percent = self.config.CURRICULUM.START_PERCENT,
                increase_interval = self.config.CURRICULUM.INCREASE_INTERVAL,
                increase_factor = self.config.CURRICULUM.INCREASE_FACTOR
            )
            train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler)
        else:
            raise NotImplementedError('Invalid order filename')
        
        return train_examples, train_features, train_dataset, train_dataloader

class CurriculumBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last,
                 total_iterations, start_percent, increase_interval, increase_factor):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.total_iterations = total_iterations

        self.N = len(self.data_source)
        self.full_data_index = list(range(self.N))

        self.target_list = [ ]
        cur_percent = start_percent
        for batch_index in range(self.total_iterations):
            if batch_index and not batch_index % increase_interval:
                cur_percent = min(cur_percent * increase_factor, 1)
            limit = np.int(np.ceil(self.N * cur_percent))
            cur_data_index = self.full_data_index[:limit]
            target_batch_index = np.random.choice(cur_data_index, self.batch_size, replace = False)
            self.target_list.append(target_batch_index)

    def __iter__(self):
        return iter(self.target_list)

    def __len__(self):
        return self.total_iterations