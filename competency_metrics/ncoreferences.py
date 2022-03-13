import os
import sys
import json
import re
import string
from functools import reduce

from tqdm import tqdm
from nltk import pos_tag, word_tokenize

# personal pronouns
pronouns = """i you he she we they it her his mine my our ours their thy your hers herself him himself hisself itself me myself one oneself ours ourselves ownself self thee theirs them themselves thou thy us""".split()

def main():
    assert len(sys.argv) == 2, 'python ncoreferences.py /path/to/dump.json'
    dump_json_filename = sys.argv[1]
    target_dirname = os.path.dirname(dump_json_filename)

    with open(dump_json_filename) as f:
        dump_json = json.load(f)

    ncoreferences_list = { }

    for i, (qid, item) in enumerate(tqdm(dump_json.items())):
        question = item['question'] + item['context']
        question_tokens = word_tokenize(question)
        question_pos = pos_tag(question_tokens)
        coreferences_cache = [ ]
        for tok, pos in question_pos:
            if tok in pronouns and pos in { 'PRP', 'PRP$' }:
                coreferences_cache.append(tok)
        ncoreferences_list[qid] = { 'value': len(coreferences_cache) / len(question_tokens), 'details': coreferences_cache }

    with open(os.path.join(target_dirname, 'ncoreferences.json'), 'w') as f:
        json.dump(ncoreferences_list, f, indent = 4)

if __name__ == '__main__':
    main()