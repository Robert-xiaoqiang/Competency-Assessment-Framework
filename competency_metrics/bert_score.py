import sys
del sys.path[0]
import os
import json
from tqdm import tqdm
from bert_score import score

def bert_score_sentence_level(hyp, ref):
    P, R, F1 = score([ hyp ], [ ref ], lang='en')

    return F1.detach().cpu().numpy().tolist()

def bert_score_corpus_level(hyps, refs):
    P, R, F1 = score(hyps, refs, lang='en')

    return F1.detach().cpu().numpy().tolist()

def main():
    with open(dump_filename) as f:
        dump_list = eval(f.read())
    
    qgs = [ ]
    refs = [ ]

    for i, item in enumerate(dump_list):
        qg, ref = item['qg'], item['question']
        qgs.append(' '.join(qg))
        refs.append(' '.join(ref))

    bert_score_list = bert_score_corpus_level(qgs, refs)

    with open('bert_score.list', 'w') as f:
        print(bert_score_list, file = f)

def main_relatedness():
    assert len(sys.argv) == 2, 'python bert_score.py /path/to/dump.json'
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

    # bert_score_relatedness_list = [ ]
    # chunk_size = 200
    # for begin in range(0, len(contexts), chunk_size):
    #     chunked_contexts, chunked_refs = contexts[begin:begin + chunk_size], refs[begin:begin + chunk_size]
    #     chunked_score = bert_score_corpus_level(chunked_contexts, chunked_refs)
    #     bert_score_relatedness_list.extend(chunked_score)
    #     print('chunked')

    score_list = bert_score_corpus_level(contexts, refs)
    bert_score_relatedness_list = { 
        qid: { 'value': score }
        for qid, score in zip(id_list, score_list)
    }
    with open(os.path.join(target_dirname, 'bert_score_relatedness.json'), 'w') as f:
        json.dump(bert_score_relatedness_list, f, indent = 4)

if __name__ == '__main__':
    main_relatedness()