import os
import sys
import json
import re
import string
from functools import reduce

from tqdm import tqdm
from nltk.corpus import stopwords
from nltk import word_tokenize

stops = stopwords.words("english")

interrogatives = ["what", "when", "where", "who", "which", "why", "whom", "how"]
question_stops = list(set(stops) - set(interrogatives))

def main():
    assert len(sys.argv) == 2, 'python nstopwords.py /path/to/dump.json'
    dump_json_filename = sys.argv[1]
    target_dirname = os.path.dirname(dump_json_filename)

    with open(dump_json_filename) as f:
        dump_json = json.load(f)

    nstopwords_list = { }

    for i, (qid, item) in enumerate(tqdm(dump_json.items())):
        question = item['question'] + item['context']
        question_tokens = word_tokenize(question)
        stopwords_cache = [ ]
        for tok in question_tokens:
            if tok in question_stops:
                stopwords_cache.append(tok)
        nstopwords_list[qid] = { 'value': len(stopwords_cache) / len(question_tokens), 'details': stopwords_cache }

    with open(os.path.join(target_dirname, 'nstopwords.json'), 'w') as f:
        json.dump(nstopwords_list, f, indent = 4)

if __name__ == '__main__':
    main()