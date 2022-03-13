import os
import sys
import json
import re
import string
from functools import reduce

from tqdm import tqdm
from nltk import word_tokenize, pos_tag

def main():
    assert len(sys.argv) == 2, 'python ngenitive.py /path/to/dump.json'
    dump_json_filename = sys.argv[1]
    target_dirname = os.path.dirname(dump_json_filename)

    with open(dump_json_filename) as f:
        dump_json = json.load(f)

    ngenitive_list = { }

    for i, (qid, item) in enumerate(tqdm(dump_json.items())):
        question = item['question'] + item['context']
        question_tokens = word_tokenize(question)
        question_pos = pos_tag(question_tokens)
        genitive_cache = [ ]
        for tok, pos in question_pos:
            # 's 's 's 's
            # if pos in { 'POS', 'WP$' } or \
            if tok.find("'s") != -1 or \
               tok == 'of':
                genitive_cache.append(tok)
        ngenitive_list[qid] = { 'value': len(genitive_cache) / len(question_tokens), 'details': genitive_cache }

    with open(os.path.join(target_dirname, 'ngenitive.json'), 'w') as f:
        json.dump(ngenitive_list, f, indent = 4)

if __name__ == '__main__':
    main()