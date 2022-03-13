import os
import sys
import json
import re
import string
from functools import reduce

from tqdm import tqdm

from linguistics_utils import get_tokens

junction_words = "not n't no nor and or".split()

def main():
    assert len(sys.argv) == 2, 'python njunctions.py /path/to/dump.json'
    dump_json_filename = sys.argv[1]
    target_dirname = os.path.dirname(dump_json_filename)

    with open(dump_json_filename) as f:
        dump_json = json.load(f)

    njunctions_list = { }

    for i, (qid, item) in enumerate(tqdm(dump_json.items())):
        question = item['question'] + item['context']
        question_tokens = get_tokens(question)
        junctions_cache = [ ]
        for tok in question_tokens:
            if tok in junction_words:
                junctions_cache.append(tok)
        njunctions_list[qid] = { 'value': len(junctions_cache) / len(question_tokens), 'details': junctions_cache }

    with open(os.path.join(target_dirname, 'njunctions.json'), 'w') as f:
        json.dump(njunctions_list, f, indent = 4)

if __name__ == '__main__':
    main()