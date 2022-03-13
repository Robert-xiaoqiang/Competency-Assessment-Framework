import json
import os
from functools import partial
from multiprocessing import Pool, cpu_count
from enum import Enum

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.utils.data import TensorDataset
import numpy as np
from tqdm import tqdm

from .SquadProcessor import SquadProcessor, SquadConverter

class HotpotProcessor(SquadProcessor):
    def __init__(self):
        pass

    def get_train_examples(self, train_filename):
        with open(
            train_filename, "r", encoding="utf-8"
        ) as reader:
            hotpot_dict = json.load(reader)
        squad_dict = self.convert_hotpot_dict_to_squad_dict(hotpot_dict)["data"]
        return self._create_examples(squad_dict, "train")

    def get_dev_examples(self, dev_filename):
        with open(
            dev_filename, "r", encoding="utf-8"
        ) as reader:
            hotpot_dict = json.load(reader) #[:9216]
        squad_dict = self.convert_hotpot_dict_to_squad_dict(hotpot_dict)["data"]
        return self._create_examples(squad_dict, "dev")

    def convert_hotpot_dict_to_squad_dict(self, hotpot_dict, gold_paras_only=True, combine_context=True):
        new_dict = { "data": [ ] }
        count = 0
        for example in hotpot_dict:
            raw_contexts = example["context"]

            if gold_paras_only:
                support = {
                    para_title: line_num
                    for para_title, line_num in example["supporting_facts"]
                }
                raw_contexts = [lst for lst in raw_contexts if lst[0] in support]

            contexts = ["".join(lst[1]) for lst in raw_contexts]
            if combine_context:
                contexts = [" ".join(contexts)]

            answer = example["answer"]
            for context in contexts:
                context = self._add_yes_no(context)
                answer_start = context.index(answer) if answer in context else -1

                new_dict["data"].append(
                    self._create_para_dict(
                        self._create_example_dict(
                            context=context,
                            answer_start=answer_start,
                            answer=answer,
                            id=example["_id"],
                            is_impossible=(answer_start == -1),
                            question=example["question"],
                        )
                    )
                )
                count += 1
        return new_dict

    def _create_example_dict(self, context, answer_start, answer, id, is_impossible, question):
        return {
            "context": context,
            "qas": [
                {
                    "answers": [{"answer_start": answer_start, "text": answer}],
                    "id": id,
                    "is_impossible": is_impossible,
                    "question": question,
                }
            ],
        }

    def _create_para_dict(self, example_dicts):
        # either multiple (context, q) dicts from multiple documents
        # or only one (context, q) dict from the join of documents
        if type(example_dicts) == dict:
            example_dicts = [example_dicts]
        return { "title": "a hotpot fake title",
                 "paragraphs": example_dicts }

    def _add_yes_no(self, context):
        # Allow model to explicitly select yes/no from text (location front, avoid truncation)
        return " ".join(["yes", "no", context])

HotpotConverter = SquadConverter