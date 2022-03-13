import json
import os
import sys
import random
random.seed(32767)

import numpy as np
np.random.seed(32767)

from tqdm import tqdm

source_dict = {
    'reading_words.json': [ 'intra1.json', 'intra2.json', 'intra3.json', 'intra4.json', 'entropy1.json', 'entropy2.json', 'nstopwords.json' ],

    'reading_sentences.json': [ 'ngenitive.json', 'readability_flesch_kincaid.json', 'readability_flesch.json', 'readability_gunning_fog.json', 'readability_coleman_liau.json', 'readability_dale_chall.json', 'readability_ari.json', 'readability_linsear_write.json', 'readability_spache.json' ], #, 'tree_height.json', 'tree_nodes.json', 'tree_pos_types.json', 'tree_width.json' ],

    'understanding_words.json': [ 'nnumerics.json', 'nlogicals.json' ],

    'understanding_sentences.json': [ 'bleu1_overlap.json', 'bert_score_relatedness.json', 'moverscore_relatedness.json', 'bleurt_relatedness.json', 'ncoreferences.json', 'njunctions.json', 'ncausals.json', 'nspatialtemporals.json' ], #, 'nfacts.json' ],

    'vocabulary.json': [ 'intra1.json', 'intra2.json', 'intra3.json', 'intra4.json', 'entropy1.json', 'entropy2.json' ],
    
    'special_words.json': [ 'nstopwords.json' ],
    
    'grammaticality.json': [ 'ngenitive.json' ], #, 'tree_height.json', 'tree_nodes.json', 'tree_pos_types.json', 'tree_width.json' ],
    
    'readability.json': [ 'readability_flesch_kincaid.json', 'readability_flesch.json', 'readability_gunning_fog.json', 'readability_coleman_liau.json', 'readability_dale_chall.json', 'readability_ari.json', 'readability_linsear_write.json', 'readability_spache.json' ],

    'words_linguistic_reasoning.json': [ 'nnumerics.json' ],
    
    'words_factual_reasoning.json': [ 'nlogicals.json' ],

    'sentences_linguistic_reasoning.json': [ 'bleu1_overlap.json', 'bert_score_relatedness.json', 'moverscore_relatedness.json', 'bleurt_relatedness.json', 'ncoreferences.json', 'njunctions.json' ],

    'sentences_factual_reasoning.json': [ 'ncausals.json', 'nspatialtemporals.json' ], #, 'nfacts.json' ],

    'merge_all.json': [ 'intra1.json', 'intra2.json', 'intra3.json', 'intra4.json', 'entropy1.json', 'entropy2.json', 'nstopwords.json', 'ngenitive.json', 'readability_flesch_kincaid.json', 'readability_flesch.json', 'readability_gunning_fog.json', 'readability_coleman_liau.json', 'readability_dale_chall.json', 'readability_ari.json', 'readability_linsear_write.json', 'readability_spache.json', 'nnumerics.json',
    # 'tree_height.json', 'tree_nodes.json', 'tree_pos_types.json', 'tree_width.json',
    'nlogicals.json', 'bleu1_overlap.json', 'bert_score_relatedness.json', 'moverscore_relatedness.json', 'bleurt_relatedness.json', 'ncoreferences.json', 'njunctions.json', 'ncausals.json', 'nspatialtemporals.json' ], #, 'nfacts.json' ]
}

def main():
    assert len(sys.argv) == 2, 'python difficulty_merger.py /path/to/evaluation_dataset_root'
    source_dirname = sys.argv[1]
    target_dirname = os.path.join(source_dirname, 'difficulty')
    keyed_target_dirname = os.path.join(source_dirname, 'keyed_difficulty')
    os.makedirs(target_dirname, exist_ok = True)
    os.makedirs(keyed_target_dirname, exist_ok = True)
    tqdm_iter = tqdm(source_dict.items())
    for merged_filename, dependent_list in tqdm_iter:
        tqdm_iter.set_description(merged_filename)
        N = len(dependent_list)
        # list of dictionaries
        factor_list = list(map(lambda f: json.load(open(os.path.join(source_dirname, f))), dependent_list))
        factor_keys = factor_list[0].keys()
        merged_scores = [ ]
        for factor_key in factor_keys:
            cur_score_list = [ ]
            # len(factor_list) == len(dependent_list)
            for factor_dict in factor_list:
                cur_score_list.append(factor_dict[factor_key]['maxmin_difficulty_score'])
            # merger algorithm (scaling summation for cdf-difficulty, direct summation for maxmin-difficulty)
            cur_score = sum(cur_score_list) / N
            # merged_scores corespond to factor_keys directly !!!!
            merged_scores.append(cur_score)
        
        target_list = list(sorted(zip(factor_keys, merged_scores), key = lambda t: t[1]))
        target_dict = dict(zip(factor_keys, merged_scores))

        with open(os.path.join(target_dirname, merged_filename), 'w') as f:
            json.dump(target_list, f, indent = 4)

        with open(os.path.join(keyed_target_dirname, merged_filename), 'w') as f:
            json.dump(target_dict, f, indent = 4)

if __name__ == '__main__':
    main()