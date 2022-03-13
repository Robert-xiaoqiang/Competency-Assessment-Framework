# Use the original version with BERTMNLI to reproduce the results.
# from moverscore import get_idf_dict, word_mover_score
# Recommend to use this version (DistilBERT) for evaluation, if the speed is your concern.
# from moverscore_v2 import get_idf_dict, word_mover_score 
import sys
del sys.path[0]
import os
os.environ['MOVERSCORE'] = '/home/xqwang/projects/qgqa/evaluation/moverscore_checkpoint'
import json
from tqdm import tqdm, trange
from moverscore import get_idf_dict, word_mover_score

def moverscore_corpus_level(hyps, refs):
    idf_dict_hyp = get_idf_dict(hyps)
    idf_dict_ref = get_idf_dict(refs)
    scores = word_mover_score(refs, hyps, idf_dict_ref, idf_dict_hyp,
                              stop_words=[], n_gram=1, remove_subwords=True)

    return scores

def main():
    with open(dump_filename) as f:
        dump_list = eval(f.read())

        qgs = [ ]
        refs = [ ]

        for i, item in enumerate(dump_list):
            # reference-free
            qg, ref = item['qg'], item['question']
            qgs.append(' '.join(qg))
            refs.append(' '.join(ref))

    moverscore_list = moverscore_corpus_level(qgs, refs)

    with open('moverscore.list', 'w') as f:
        print(moverscore_list, file = f)

def main_relatedness():
    assert len(sys.argv) == 2, 'python moverscore.py /path/to/dump.json'
    dump_json_filename = sys.argv[1]
    target_dirname = os.path.dirname(dump_json_filename)

    with open(dump_json_filename) as f:
        dump_json = json.load(f)

    contexts = [ ]
    refs = [ ]
    id_list = [ ]

    for i, (qid, item) in enumerate(dump_json.items()):
        # reference-free
        context, ref = item['context'], item['question']
        contexts.append(context[:512]) # ~100 words
        refs.append(ref)
        id_list.append(qid)

    score_list = [ ]
    chunk_size = 100
    for begin in trange(0, len(contexts), chunk_size):
        chunked_contexts, chunked_refs = contexts[begin:begin + chunk_size], refs[begin:begin + chunk_size]
        chunked_score = moverscore_corpus_level(chunked_contexts, chunked_refs)
        score_list.extend(chunked_score)
        print('chunked')

    moverscore_relatedness_list = { 
        qid: { 'value': score }
        for qid, score in zip(id_list, score_list)
    }

    with open(os.path.join(target_dirname, 'moverscore_relatedness.json'), 'w') as f:
        json.dump(moverscore_relatedness_list, f, indent = 4)

if __name__ == '__main__':
    main_relatedness()