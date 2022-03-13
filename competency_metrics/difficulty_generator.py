import json
import os
import sys
import random
random.seed(32767)

import numpy as np
np.random.seed(32767)
import pandas as pd

from tqdm import tqdm

source_filenames = [
    'intra1.json', 'intra2.json', 'intra3.json', 'intra4.json',
    'entropy1.json', 'entropy2.json', 'nstopwords.json',
    
    'ngenitive.json',
    'tree_height.json', 'tree_nodes.json', 'tree_pos_types.json', 'tree_width.json',
    'readability_flesch_kincaid.json', 'readability_flesch.json', 'readability_gunning_fog.json', 'readability_coleman_liau.json',
    'readability_dale_chall.json', 'readability_ari.json', 'readability_linsear_write.json', 'readability_spache.json',

    'nnumerics.json', 'nlogicals.json',

    'bleu1_overlap.json',
    'bert_score_relatedness.json', 'moverscore_relatedness.json', 'bleurt_relatedness.json', 
    'ncoreferences.json', 'njunctions.json',

    'ncausals.json', 'nspatialtemporals.json', 'nfacts.json'
]

def main():
    assert len(sys.argv) == 2, 'python difficulty_generator.py /path/to/evaluation_dataset_root'
    source_dirname = sys.argv[1]

    with open(os.path.join(source_dirname, 'prediction.json')) as f:
        prediction_dict = json.load(f)
    prediction_list = np.asarray(list(map(lambda d: d['f1_score'] * d['start_prob'] * d['end_prob'], prediction_dict.values())))

    source_identifiers = [ ]
    for filename in source_filenames:
        with open(os.path.join(source_dirname, filename)) as f:
            index = filename.index('.json')
            dynamic_identifier = filename[:index] + '_dict'
            dynamic_statements = dynamic_identifier + ' = json.load(f)'
            exec(dynamic_statements)
            source_identifiers.append(eval(dynamic_identifier))
    tqdm_iter = tqdm(zip(source_identifiers, source_filenames, source_pn_list), total = len(source_identifiers))
    for factor, filename, pn in tqdm_iter:
        tqdm_iter.set_description(filename)
        values = list(map(lambda d: d['value'], factor.values()))
        max_score = max(values)
        bins = 50 if max_score <= 1 else 500
        hist, hist_edges = np.histogram(list(values), bins = bins)
        pdf = hist / hist.sum()
        cdf = pdf.cumsum()
        for key, value_dict in factor.items():
            for lower_index, lower, upper in zip(range(len(hist_edges[:-1])), hist_edges[:-1], hist_edges[1:]):
                if lower <= value_dict['value'] <= upper:
                    cdf_score = cdf[lower_index]
                    difficulty_score = cdf_score if pn == 1 else 1 - cdf_score
                    break
            factor[key].update({
                'cdf_score': cdf_score,
                'difficulty_score': difficulty_score
            })

        with open(os.path.join(source_dirname, filename), 'w') as f:
            json.dump(factor, f, indent = 4)

def main_maxmin():
    assert len(sys.argv) == 2, 'python difficulty_generator.py /path/to/evaluation_dataset_root'
    source_dirname = sys.argv[1]

    with open(os.path.join(source_dirname, 'prediction.json')) as f:
        prediction_dict = json.load(f)
    prediction_list = list(map(lambda d: d['f1_score'] * d['start_prob'] * d['end_prob'], prediction_dict.values()))

    source_identifiers = [ ]
    df_source = {
        'prediction': prediction_list
    }
    for filename in source_filenames:
        with open(os.path.join(source_dirname, filename)) as f:
            index = filename.index('.json')
            dynamic_identifier = filename[:index] + '_dict'
            dynamic_statements = dynamic_identifier + ' = json.load(f)'
            exec(dynamic_statements)
            source_identifiers.append(eval(dynamic_identifier))
            df_source[dynamic_identifier] = list(map(lambda d: d['value'], (eval(dynamic_identifier)).values()))
    
    df = pd.DataFrame.from_dict(df_source)
    corr = df.corr()
    source_pn_list = np.sign(corr.loc['prediction'].to_numpy() * (-1)).tolist()[1:]
    dynamic_identifiers = corr.columns.tolist()[1:]

    print(list(zip(dynamic_identifiers, source_pn_list)))

    tqdm_iter = tqdm(zip(dynamic_identifiers, source_pn_list), total = len(source_identifiers))
    for factor_dynamic_identifier, pn in tqdm_iter:
        factor = eval(factor_dynamic_identifier)
        index = factor_dynamic_identifier.index('_dict')
        filename = factor_dynamic_identifier[:index] + '.json'
        tqdm_iter.set_description(filename)
        values = list(map(lambda d: d['value'], factor.values()))
        max_score, min_score = max(values), min(values)
        for key, value_dict in factor.items():
            maxmin_difficulty_score = (value_dict['value'] - min_score) / (max_score - min_score)
            maxmin_difficulty_score = maxmin_difficulty_score if pn > 0 else 1 - maxmin_difficulty_score
            factor[key].update({
                'maxmin_difficulty_score': maxmin_difficulty_score
            })

        with open(os.path.join(source_dirname, filename), 'w') as f:
            json.dump(factor, f, indent = 4)

if __name__ == '__main__':
    main()