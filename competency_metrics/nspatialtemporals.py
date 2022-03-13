import os
import sys
import json
import re
import string
from functools import reduce

from tqdm import tqdm
from nltk import pos_tag

from linguistics_utils import get_tokens

temporal_words = "when while as whenever before after since until till soon moment minute second time".split()
spatial_words = "at in on above over below under front behind".split()

def main():
    assert len(sys.argv) == 2, 'python nspatialtemporals.py /path/to/dump.json'
    dump_json_filename = sys.argv[1]
    target_dirname = os.path.dirname(dump_json_filename)

    with open(dump_json_filename) as f:
        dump_json = json.load(f)

    nspatialtemporals_list = { }

    for i, (qid, item) in enumerate(tqdm(dump_json.items())):
        question = item['question'] + item['context']
        question_tokens = get_tokens(question)
        question_pos = pos_tag(question_tokens)
        spatialtemporals_cache = [ ]
        for tok, pos in question_pos:
            if tok in spatial_words or tok in temporal_words:
                spatialtemporals_cache.append(pos)
        nspatialtemporals_list[qid] = { 'value': len(spatialtemporals_cache) / len(question_tokens), 'details': spatialtemporals_cache }

    with open(os.path.join(target_dirname, 'nspatialtemporals.json'), 'w') as f:
        json.dump(nspatialtemporals_list, f, indent = 4)

if __name__ == '__main__':
    main()