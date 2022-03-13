import os
import sys
import json
import re
import string
from functools import reduce

from tqdm import tqdm
from nltk import pos_tag

from linguistics_utils import get_tokens

causal_words = "because why therefore cause reason as since due owing consequence consequent consequently thus hence account".split()

def main():
    assert len(sys.argv) == 2, 'python ncausals.py /path/to/dump.json'
    dump_json_filename = sys.argv[1]
    target_dirname = os.path.dirname(dump_json_filename)

    with open(dump_json_filename) as f:
        dump_json = json.load(f)

    ncausals_list = { }

    for i, (qid, item) in enumerate(tqdm(dump_json.items())):
        question = item['question'] + item['context']
        question_tokens = get_tokens(question)
        question_pos = pos_tag(question_tokens)
        causals_cache = [ ]
        for tok, pos in question_pos:
            if tok in causal_words:
                causals_cache.append(tok)
        ncausals_list[qid] = { 'value': len(causals_cache) / len(question_tokens), 'details': causals_cache }

    with open(os.path.join(target_dirname, 'ncausals.json'), 'w') as f:
        json.dump(ncausals_list, f, indent = 4)

if __name__ == '__main__':
    main()