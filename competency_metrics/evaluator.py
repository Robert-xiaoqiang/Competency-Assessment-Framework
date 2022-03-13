import os
import sys
import json
import re
import string

from tqdm import tqdm
from nltk import word_tokenize

from distinct import distinct_n_sentence_level
from rouge import rouge_n_sentence_level
from bleu import bleu_sentence_level
from embedding import embedding_sentence_level
from entropy import entropy_sentence_level

def main():
    assert len(sys.argv) == 2, 'python evaluator.py /path/to/dump.json'
    dump_json_filename = sys.argv[1]
    target_dirname = os.path.dirname(dump_json_filename)

    with open(dump_json_filename) as f:
        dump_json = json.load(f)

    intra1_list = { }
    intra2_list = { }
    intra3_list = { }
    intra4_list = { }

    bleu1_overlap_list = { }
    bleu2_overlap_list = { }
    bleu3_overlap_list = { }
    bleu4_overlap_list = { }

    emb_avg_list = { }
    emb_extrema_list = { }
    emb_greedy_list = { }

    entropy1_list = { }
    entropy2_list = { }

    for i, qid in enumerate(tqdm(dump_json)):
        # reference-free
        item = dump_json[qid]
        ref = item['question']
        context = item['context']

        tokenized_context = word_tokenize(context)
        tokenized_ref = word_tokenize(ref)
        tokenized_text = tokenized_context + tokenized_ref

        intra1 = distinct_n_sentence_level(tokenized_text, 1)
        intra2 = distinct_n_sentence_level(tokenized_text, 2)
        intra3 = distinct_n_sentence_level(tokenized_text, 3)
        intra4 = distinct_n_sentence_level(tokenized_text, 4)
        
        bleu1 = bleu_sentence_level(tokenized_context, tokenized_ref, max_order = 1).precisions[0]
        bleu2 = bleu_sentence_level(tokenized_context, tokenized_ref, max_order = 2).precisions[1]
        bleu3 = bleu_sentence_level(tokenized_context, tokenized_ref, max_order = 3).precisions[2]
        bleu4 = bleu_sentence_level(tokenized_context, tokenized_ref, max_order = 4).precisions[3]

        # emb_avg, emb_extrema, emb_greedy = embedding_sentence_level(context, ref)

        entropy1, entropy2 = entropy_sentence_level(tokenized_text)

        intra1_list[qid] = { 'value': intra1 }
        intra2_list[qid] = { 'value': intra2 }
        intra3_list[qid] = { 'value': intra3 }
        intra4_list[qid] = { 'value': intra4 }

        bleu1_overlap_list[qid] = { 'value': bleu1 }
        bleu2_overlap_list[qid] = { 'value': bleu2 }
        bleu3_overlap_list[qid] = { 'value': bleu3 }
        bleu4_overlap_list[qid] = { 'value': bleu4 }

        # emb_avg_list[qid] = { 'value': float(emb_avg) }
        # emb_extrema_list[qid] = { 'value': float(emb_extrema) }
        # emb_greedy_list[qid] = { 'value': float(emb_greedy) }

        entropy1_list[qid] = { 'value': entropy1 }
        entropy2_list[qid] = { 'value': entropy2 }        

    with open(os.path.join(target_dirname, 'intra1.json'), 'w') as f:
        json.dump(intra1_list, f, indent = 4)
    with open(os.path.join(target_dirname, 'intra2.json'), 'w') as f:
        json.dump(intra2_list, f, indent = 4)
    with open(os.path.join(target_dirname, 'intra3.json'), 'w') as f:
        json.dump(intra3_list, f, indent = 4)
    with open(os.path.join(target_dirname, 'intra4.json'), 'w') as f:
        json.dump(intra4_list, f, indent = 4)

    with open(os.path.join(target_dirname, 'bleu1_overlap.json'), 'w') as f:
        json.dump(bleu1_overlap_list, f, indent = 4)
    with open(os.path.join(target_dirname, 'bleu2_overlap.json'), 'w') as f:
        json.dump(bleu2_overlap_list, f, indent = 4)
    with open(os.path.join(target_dirname, 'bleu3_overlap.json'), 'w') as f:
        json.dump(bleu3_overlap_list, f, indent = 4)
    with open(os.path.join(target_dirname, 'bleu4_overlap.json'), 'w') as f:
        json.dump(bleu4_overlap_list, f, indent = 4)

    # with open(os.path.join(target_dirname, 'emb_avg.json'), 'w') as f:
    #     json.dump(emb_avg_list, f, indent = 4)
    # with open(os.path.join(target_dirname, 'emb_extrema.json'), 'w') as f:
    #     json.dump(emb_extrema_list, f, indent = 4)
    # with open(os.path.join(target_dirname, 'emb_greedy.json'), 'w') as f:
    #     json.dump(emb_greedy_list, f, indent = 4)

    with open(os.path.join(target_dirname, 'entropy1.json'), 'w') as f:
        json.dump(entropy1_list, f, indent = 4)
    with open(os.path.join(target_dirname, 'entropy2.json'), 'w') as f:
        json.dump(entropy2_list, f, indent = 4)

if __name__ == '__main__':
    main()