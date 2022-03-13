#! /usr/bin/bash
DUMP="squad1_devtoy/dump.json"

python evaluator.py $DUMP & \
python nstopwords.py $DUMP & \
python ngenitive.py $DUMP & \
python tree.py $DUMP & \
python readability.py $DUMP & \
python nnumerics.py $DUMP & \
python nlogicals.py $DUMP & \
python bert_score.py $DUMP & \
python moverscore.py $DUMP & \
python bleurt.py $DUMP & \
python ncoreferences.py $DUMP & \
python njunctions.py $DUMP & \
python ncausals.py $DUMP & \
python nspatialtemporals.py $DUMP & \
python nfacts.py $DUMP