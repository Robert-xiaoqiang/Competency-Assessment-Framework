import sys
del sys.path[0]
import os
import json
from tqdm import tqdm
from nltk import word_tokenize
from readability import Readability

def readability_sentence_level(text):
    r = Readability(text)
    ret = [ r.flesch_kincaid().score,
            r.flesch().score,
            r.gunning_fog().score,
            r.coleman_liau().score,
            
            r.dale_chall().score,
            r.ari().score,
            r.linsear_write().score,
            # r.smog(True).score,
            r.spache().score
    ]
    return ret

target_identifiers = [
    'readability_flesch_kincaid',
    'readability_flesch',
    'readability_gunning_fog',
    'readability_coleman_liau',
    'readability_dale_chall',
    'readability_ari',
    'readability_linsear_write',
    'readability_spache'
]

def main():
    assert len(sys.argv) == 2, 'python readability.py /path/to/dump.json'
    dump_json_filename = sys.argv[1]
    target_dirname = os.path.dirname(dump_json_filename)

    with open(dump_json_filename) as f:
        dump_json = json.load(f)
    target_variables = [ ]
    for name in target_identifiers:
        dynamic_statement = name + ' = { }'
        exec(dynamic_statement)
        target_variables.append(eval(name))

    for i, (qid, item) in enumerate(tqdm(dump_json.items())):
        text = item['question'] + item['context']
        n_tokens = len(word_tokenize(text))
        if n_tokens <= 150:
            text = (text + ' ') * (100 // n_tokens + 2) # maybe punctuation
        readability_score = readability_sentence_level(text)
        for rid, variable in enumerate(target_variables):
            variable[qid] = { 'value': readability_score[rid] }
    for name in target_identifiers:
        filename = name + '.json'
        with open(os.path.join(target_dirname, filename), 'w') as f:
            json.dump(eval(name), f, indent = 4)

if __name__ == '__main__':
    main()
