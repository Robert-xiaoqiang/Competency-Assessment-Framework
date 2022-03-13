import sys
del sys.path[0]
import os
import json
from tqdm import tqdm
from bleurt import score
checkpoint = 'bleurt_checkpoint/bleurt-base-128'

scorer = score.BleurtScorer(checkpoint)

def bleurt_corpus_level(hyp, ref):
    scores = scorer.score(ref, hyp)

    return scores

def main():
    with open(dump_filename) as f:
        dump_list = eval(f.read())

    bleurt_list = [ ]

    for i, item in enumerate(dump_list):
        # reference-free
        qg, ref = item['qg'], item['question']

        bleurt = bleurt_sentence_level(qg, ref)
        bleurt_list.append(bleurt)
        print(i)

    with open('bleurt.list', 'w') as f:
        print(bleurt_list, file = f)

def main_relatedness():
    assert len(sys.argv) == 2, 'python bleurt.py /path/to/dump.json'
    dump_json_filename = sys.argv[1]
    target_dirname = os.path.dirname(dump_json_filename)

    with open(dump_json_filename) as f:
        dump_json = json.load(f)

    contexts = [ ]
    refs = [ ]
    id_list = [ ]

    for i, (qid, item) in enumerate(dump_json.items()):
        context, ref = item['context'], item['question']
        contexts.append(context)
        refs.append(ref)
        id_list.append(qid)

    score_list = bleurt_corpus_level(contexts, refs)
    bleurt_relatedness_list = { 
        qid: { 'value': score }
        for qid, score in zip(id_list, score_list)
    }

    with open(os.path.join(target_dirname, 'bleurt_relatedness.json'), 'w') as f:
        json.dump(bleurt_relatedness_list, f, indent = 4)

if __name__ == '__main__':
    main_relatedness()