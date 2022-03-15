import os
import json
from tqdm import tqdm
dataset_filename = '/home/xqwang/projects/qgqa/squad2/train-v2.0.json'
dump_json_filename = 'dump.json'

def main():
    with open(dataset_filename) as f:
        squad_dict = json.load(f)
    '''
    'xxx': {
        question: '',
        answer: '' | [],
        context: ''
    }
    '''
    dump_json = { }
    for doc in tqdm(squad_dict['data']):
        title = doc['title']
        for paragraph in doc["paragraphs"]:
            context_text = paragraph["context"]
            for qa in paragraph["qas"]:
                qid = qa["id"]
                question_text = qa["question"]

                is_impossible = qa.get("is_impossible", False)
                # plausible answers ????

                answer_text = list(map(lambda a: a['text'], qa["answers"]))
                dump_json[qid] = {
                    'question': question_text,
                    'is_impossible': is_impossible,
                    'answer': answer_text,
                    'context': context_text
                }
    with open(dump_json_filename, 'w') as f:
        json.dump(dump_json, f, indent = 4)

if __name__ == '__main__':
    main()
