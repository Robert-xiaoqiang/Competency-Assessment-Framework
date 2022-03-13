import os
import sys
import json
import re
import string
from functools import reduce

from tqdm import tqdm
from nltk import pos_tag

from linguistics_utils import get_tokens

def has_digits(s):
    return any(char.isdigit() for char in s)

def main():
    assert len(sys.argv) == 2, 'python nnumerics.py /path/to/dump.json'
    dump_json_filename = sys.argv[1]
    target_dirname = os.path.dirname(dump_json_filename)

    with open(dump_json_filename) as f:
        dump_json = json.load(f)

    nnumerics_list = { }

    for i, (qid, item) in enumerate(tqdm(dump_json.items())):
        question = item['question'] + item['context']
        question_tokens = get_tokens(question)
        question_pos = pos_tag(question_tokens)
        numerics_count = 0
        last_pos_cd = False
        for tok, pos in question_pos:
            if pos == 'CD' and has_digits(tok):
                last_pos_cd = True
            elif pos in { 'NN', 'NNS', 'NNP'}:
                if last_pos_cd:
                    numerics_count += 1
                last_pos_cd = False
            else:
                last_pos_cd = False
        nnumerics_list[qid] = { 'value': numerics_count / len(question_tokens) }

    with open(os.path.join(target_dirname, 'nnumerics.json'), 'w') as f:
        json.dump(nnumerics_list, f, indent = 4)

if __name__ == '__main__':
    main()