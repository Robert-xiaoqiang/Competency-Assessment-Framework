import os
import sys
import json
from tqdm import tqdm
from nltk.parse.corenlp import CoreNLPParser, CoreNLPDependencyParser
from nltk.tree import Tree

constituency_parser = CoreNLPParser()
dependency_parser = CoreNLPDependencyParser()

def count_nodes(tree):
    ret = 1 # str (not nltk.tree.Tree) is the type of terminal/leaf node
    if isinstance(tree, Tree):	
        for child in tree:
            ret += count_nodes(child)
    return ret

def constituency_tree_sentence_level(text):
    properties = { 'timeout': 300000 }
    # the most likely parsing
    tree = next(constituency_parser.raw_parse(text, properties = properties))
    height = tree.height()
    n_pos_types = len(set(map(lambda t: t[1], tree.pos()))) # also counts punctuation
    n_nodes = count_nodes(tree)
    n_leaves = len(tree.leaves())

    return [ height, n_pos_types, n_nodes, n_leaves ]

def dependency_tree_sentence_level(text):
    properties = { 'timeout': 300000 }
    # the most likely parsing
    tree = next(dependency_parser.raw_parse(text, properties = properties))
    height = tree.height()
    n_pos_types = len(set(map(lambda t: t[1], tree.pos()))) # also counts punctuation
    n_nodes = count_nodes(tree)
    n_leaves = len(tree.leaves())

    return [ height, n_pos_types, n_nodes, n_leaves ]

def main():
    assert len(sys.argv) == 2, 'python tree.py /path/to/dump.json'
    dump_json_filename = sys.argv[1]
    target_dirname = os.path.dirname(dump_json_filename)

    with open(dump_json_filename) as f:
        dump_json = json.load(f)
    
    tree_height_list = { }
    tree_pos_types_list = { }
    tree_nodes_list = { }
    tree_width_list = { }

    for i, (qid, item) in enumerate(tqdm(dump_json.items())):
        ref = item['question'][:512]

        tree_score = constituency_tree_sentence_level(ref)

        tree_height_list[qid] = { 'value': tree_score[0] }
        tree_pos_types_list[qid] = { 'value': tree_score[1] }
        tree_nodes_list[qid] = { 'value': tree_score[2] }
        tree_width_list[qid] = { 'value': tree_score[3] }

    with open(os.path.join(target_dirname, 'tree_height.json'), 'w') as f:
        json.dump(tree_height_list, f, indent = 4)
    with open(os.path.join(target_dirname, 'tree_pos_types.json'), 'w') as f:
        json.dump(tree_pos_types_list, f, indent = 4)
    with open(os.path.join(target_dirname, 'tree_nodes.json'), 'w') as f:
        json.dump(tree_nodes_list, f, indent = 4)
    with open(os.path.join(target_dirname, 'tree_width.json'), 'w') as f:
        json.dump(tree_width_list, f, indent = 4)

if __name__ == '__main__':
    main()