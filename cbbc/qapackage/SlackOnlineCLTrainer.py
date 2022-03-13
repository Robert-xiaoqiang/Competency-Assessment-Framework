import os
import json
import random
random.seed(32767)
import shutil

import numpy as np
np.random.seed(32767)

from .OnlineCLTrainer import OnlineCLTrainer

class SlackOnlineCLTrainer(OnlineCLTrainer):
    def __init__(self, model, tokenizer, train_examples, train_features, train_dataset, train_dataloader,
                 dev_examples, dev_features, dev_dataloader, config):
        super().__init__(model, tokenizer, train_examples, train_features, train_dataset, train_dataloader,
                         dev_examples, dev_features, dev_dataloader, config)

    def get_factor_threshold(self, iteration):
        dev_prediction_dirname = os.path.join(self.prediction_path, 'model_iteration_{}'.format(iteration))
        dev_prediction_file = os.path.join(dev_prediction_dirname, 'prediction.json')
        with open(dev_prediction_file) as f:
            prediction_dict = json.load(f)
        
        sorted_list = list(sorted(prediction_dict.items(), key = lambda t: t[1]['f1_score'] * t[1]['start_prob'] * t[1]['end_prob'], reverse = True))
        understood_dict = dict(sorted_list[:16])
 
        threshold_dict = { }
        for factor_entry in self.config.CURRICULUM.DEV_FACTORS:
            factor_key, factor_keyed_difficulty_filename = list(factor_entry.items())[0]
            factor_scores = 0.0
            with open(factor_keyed_difficulty_filename) as f:
                factor_keyed_difficulty_dict = json.load(f)
            for qid in understood_dict.keys():
                factor_scores += factor_keyed_difficulty_dict[qid]
            if understood_dict:
                factor_scores /= float(len(understood_dict))
            threshold_dict[factor_key] = factor_scores

        dev_factor_threshold_file = os.path.join(dev_prediction_dirname, 'factor_threshold.json')
        with open(dev_factor_threshold_file, 'w') as f:
            json.dump(threshold_dict, f, indent = 4)

        backup_dirname = os.path.join(self.prediction_path, 'model_latest')
        os.makedirs(backup_dirname, exist_ok = True)
        # for use of sampling strategy
        shutil.copy(dev_factor_threshold_file, backup_dirname)

        return threshold_dict
