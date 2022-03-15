import os
import json
from tqdm import tqdm
dataset_filename = '/home/xqwang/projects/qgqa/hotpot/hotpot_dev_distractor_v1.json'
dump_json_filename = 'dump.json'

def main(gold_paras_only=True, combine_contexts=True):
    with open(dataset_filename) as f:
        hotpot_dict = json.load(f)
    '''
    'xxx': {
        question: '',
        answer: '',
        context: '',
        supporting_facts: [ ],
        type: '',
        level: '',
    }
    '''
    dump_json = { }
    for example in tqdm(hotpot_dict):
        raw_contexts = example['context']

        if gold_paras_only:
            # support only
            support = {
                para_title: line_num
                for para_title, line_num in example['supporting_facts']
            }
            raw_contexts = [ lst for lst in raw_contexts if lst[0] in support ]
        
        # lst[1] is a context (with a sentence a str)
        contexts = [ ''.join(lst[1]) for lst in raw_contexts ]

        if combine_contexts:
            contexts = [ ' '.join(contexts) ]

        # either combine -> just one context
        # or everyone context with the original qa pair 
        for context in contexts:
            qid = example['_id']
            entry = {
                'question': example['question'],
                'answer': example['answer'],
                'context': context,
                'supporting_facts': example['supporting_facts'],
                'type': example['type'],
                'level': example['level']
            }
            dump_json[qid] = entry

    with open(dump_json_filename, 'w') as f:
        json.dump(dump_json, f, ensure_ascii = False, indent = 4)


if __name__ == '__main__':
    main()