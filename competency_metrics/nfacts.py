import os
import sys
import json
from tqdm import tqdm

def main():
    assert len(sys.argv) == 2, 'python nfacts.py /path/to/dump.json'
    dump_json_filename = sys.argv[1]
    target_dirname = os.path.dirname(dump_json_filename)

    with open(dump_json_filename) as f:
        dump_json = json.load(f)

    nfacts_list = { }
    type_list = { }
    level_list = { }
    type_dict = {
        'comparison': 0,
        'bridge': 1
    }
    level_dict = {
        'easy': 0,
        'medium': 1,
        'hard': 2
    }
    for i, (qid, item) in enumerate(tqdm(dump_json.items())):
        # reference-free
        facts = item['supporting_facts']

        nfacts = len(facts)
        nfacts_list[qid] = { 'value': nfacts }
        type_list[qid] = { 'value': type_dict[item['type']] }
        level_list[qid] = { 'value': level_dict[item['level']] }

    with open(os.path.join(target_dirname, 'nfacts.json'), 'w') as f:
        json.dump(nfacts_list, f, indent = 4)
    with open(os.path.join(target_dirname, 'type.json'), 'w') as f:
        json.dump(type_list, f, indent = 4)
    with open(os.path.join(target_dirname, 'level.json'), 'w') as f:
        json.dump(level_list, f, indent = 4)

if __name__ == '__main__':
    main()