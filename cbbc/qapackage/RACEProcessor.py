import json
import os
import logging
logger = logging.getLogger(__name__)
from functools import partial
from multiprocessing import Pool, cpu_count
from enum import Enum
import glob

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.utils.data import TensorDataset

import numpy as np
from tqdm import tqdm

class RACEExample(object):
    def __init__(self,
                 qas_id,
                 context_text,
                 question_text,
                 answer_text_0,
                 answer_text_1,
                 answer_text_2,
                 answer_text_3,
                 label = None):
        self.qas_id = qas_id
        self.context_text = context_text
        self.question_text = question_text
        self.answer_texts = [
            answer_text_0,
            answer_text_1,
            answer_text_2,
            answer_text_3,
        ]
        self.label = label

class RACEFeatures(object):
    def __init__(self,
                 qas_id,
                 example_index,
                 unique_id,
                 choices_features,
                 label

    ):
        self.qas_id = qas_id
        self.example_index = example_index
        self.unique_id = unique_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            }
            for _, input_ids, attention_mask, token_type_ids in choices_features
        ]
        self.label = label


class RACEProcessor:
    def __init__(self):
        pass

    def get_train_examples(self, train_filename):
        return self._create_examples(train_filename)

    def get_dev_examples(self, dev_filename):
        return self._create_examples(dev_filename)

    def _create_examples(self, root_dirname):
        examples = [ ]
        for question_level in [ 'middle', 'high' ]:
            dirname = os.path.join(root_dirname, question_level)
            filenames = glob.glob(dirname + "/*txt")
            for filename in tqdm(filenames, desc = 'processing {} directory'.format(question_level)):
                main_filename = os.path.splitext(os.path.basename(filename))[0]
                with open(filename) as f:
                    data_raw = json.load(f)
                    article = data_raw['article']
                    n_questions = len(data_raw['questions'])
                    for i in range(n_questions):
                        question = data_raw['questions'][i]
                        # answer label (0123 -> ABCD)
                        label = ord(data_raw['answers'][i]) - ord('A')
                        options = data_raw['options'][i]
                        # if possible(answerable)
                        # train with the first answer
                        # dev/val/test with all answers
                        examples.append(
                            RACEExample(
                                qas_id = os.path.basename(root_dirname) + '-' + question_level + '-' + main_filename + '-q' + str(i),
                                context_text = article,
                                question_text = question,

                                answer_text_0 = options[0],
                                answer_text_1 = options[1],
                                answer_text_2 = options[2],
                                answer_text_3 = options[3],
                                label = label))
                
        return examples 


class RACEConverter:
    def __init__(self):
        pass
    def convert_examples_to_features(
        self, 
        examples, 
        tokenizer, 
        max_seq_length, 
        doc_stride,
        max_query_length,
        is_training, 
        padding_strategy="max_length",
        return_dataset=False,
        threads=1,
        tqdm_enabled=True
    ):
        """Loads a data file into a list of `InputBatch`s."""
        unique_id = 1000000000
        example_index = 0
        features = []
        for example_index, example in enumerate(tqdm(examples)):
            context_tokens = tokenizer.tokenize(example.context_text)
            question_tokens = tokenizer.tokenize(example.question_text)

            choices_features = []
            for answer_index, answer in enumerate(example.answer_texts):
                qapair_tokens = question_tokens + tokenizer.tokenize(answer)
                self._truncate_seq_pair(context_tokens, qapair_tokens, max_seq_length - 3)

                tokens = ["[CLS]"] + context_tokens + ["[SEP]"] + qapair_tokens + ["[SEP]"]
                token_type_ids = [0] * (len(context_tokens) + 2) + [1] * (len(qapair_tokens) + 1)

                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                attention_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                padding = [0] * (max_seq_length - len(input_ids))
                input_ids += padding
                attention_mask += padding
                token_type_ids += padding

                assert len(input_ids) == max_seq_length
                assert len(attention_mask) == max_seq_length
                assert len(token_type_ids) == max_seq_length

                choices_features.append((tokens, input_ids, attention_mask, token_type_ids))

            label = example.label

            features.append(
                RACEFeatures(
                    qas_id = example.qas_id,
                    example_index = example_index,
                    unique_id = unique_id,
                    choices_features = choices_features,
                    label = label
                )
            )
            # identify feature
            unique_id += 1
            # identify example, here we regard an example as a feature simply
            example_index += 1

        if return_dataset == "pt":
            # Convert to Tensors and build dataset
            all_input_ids = torch.tensor([ [ choice_features['input_ids'] for choice_features in f.choices_features ]
                                           for f in features ], dtype=torch.long)
            all_attention_mask = torch.tensor([ [ choice_features['attention_mask'] for choice_features in f.choices_features ]
                                           for f in features ], dtype=torch.long)
            all_token_type_ids = torch.tensor([ [ choice_features['token_type_ids'] for choice_features in f.choices_features ]
                                           for f in features ], dtype=torch.long)
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

            if not is_training:
                all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
                dataset = TensorDataset(
                    all_input_ids,
                    all_attention_mask,
                    all_token_type_ids,
                    all_feature_index
                )
            else:
                dataset = TensorDataset(
                    all_input_ids,
                    all_attention_mask,
                    all_token_type_ids,
                    all_labels
                )
            return features, dataset
        else:
            return features

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
