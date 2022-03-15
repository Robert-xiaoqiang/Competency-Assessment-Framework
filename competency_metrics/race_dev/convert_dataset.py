import os
import json
import glob
from tqdm import tqdm
dataset_filename = '/home/xqwang/projects/qgqa/race/dev'
dump_json_filename = 'dump.json'

def main():
    '''
    Conceptually, article_id is sufficient
    'xxx': {
        context: ''
        qas: [
            {
                question: '',
                options: [ ],
                answer: ''
            },
            ...
        ]
    }
    Actually, we need question_id to maintain consistency with other datasets
    context will be broadcast into every corresponding question
    '''
    dump_json = { }
    root_dirname = dataset_filename
    for question_level in [ 'middle', 'high' ]:
        dirname = os.path.join(root_dirname, question_level)
        filenames = glob.glob(dirname + "/*txt")
        for filename in tqdm(filenames, desc = 'processing {} directory'.format(question_level)):
            main_filename = os.path.splitext(os.path.basename(filename))[0]
            with open(filename) as f:
                data_raw = json.load(f)
                article = data_raw['article']
                n_questions = len(data_raw['questions'])
                for i in range(n_questions):
                    question = data_raw['questions'][i]
                    options = data_raw['options'][i]
                    answer = data_raw['answers'][i]
                    qas_id = os.path.basename(root_dirname) + '-' + question_level + '-' + main_filename + '-q' + str(i)
                    dump_json[qas_id] = {
                        'context': article,
                        'question': question,
                        'options': dict(zip(['A', 'B', 'C', 'D'], options)),
                        'answer': answer
                    }

    with open(dump_json_filename, 'w') as f:
        json.dump(dump_json, f, indent = 4)

if __name__ == '__main__':
    main()