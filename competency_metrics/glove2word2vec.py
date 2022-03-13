# Imports
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

# Temporary file
tmp_file = get_tmpfile('temp_word2vec.txt')

# GloVe vectors loading function into temporary file
glove2word2vec('/home/xqwang/projects/qgqa/FocusSeq2Seq/glove/glove.6B.300d.txt', tmp_file)

# Creating a KeyedVectors from a temporary file
model = KeyedVectors.load_word2vec_format(tmp_file)

# Optional saving of vectors to binary format
model.save_word2vec_format('glove.6B.300d.model.bin', binary=True)