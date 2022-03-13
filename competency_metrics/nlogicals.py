import os
import sys
import json
import re
import string
from functools import reduce

from tqdm import tqdm

from linguistics_utils import get_tokens

logical_words = "all any each every few if more most other same some than".split()

def main():
    assert len(sys.argv) == 2, 'python nlogicals.py /path/to/dump.json'
    dump_json_filename = sys.argv[1]
    target_dirname = os.path.dirname(dump_json_filename)

    with open(dump_json_filename) as f:
        dump_json = json.load(f)

    nlogicals_list = { }

    for i, (qid, item) in enumerate(tqdm(dump_json.items())):
        question = item['question'] + item['context']
        question_tokens = get_tokens(question)
        logicals_cache = [ ]
        for tok in question_tokens:
            if tok in logical_words:
                logicals_cache.append(tok)
        nlogicals_list[qid] = { 'value': len(logicals_cache) / len(question_tokens), 'details': logicals_cache }

    with open(os.path.join(target_dirname, 'nlogicals.json'), 'w') as f:
        json.dump(nlogicals_list, f, indent = 4)

if __name__ == '__main__':
    main()