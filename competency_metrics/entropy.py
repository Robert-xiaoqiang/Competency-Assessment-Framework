from nltk import word_tokenize
import math

def entropy_sentence_level(tokens):  
    distro = build_distro(tokens)

    uni_entropy = sum([ -prob * math.log(prob, 2) for prob in distro['uni'].values() ])
    bi_entropy = sum([ -prob * math.log(prob, 2) for prob in distro['bi'].values() ])

    return uni_entropy, bi_entropy

# Converts frequency dict to probabilities.
def convert_to_probs(freq_dict):
  num_words = sum(list(freq_dict.values()))
  return { key: val / num_words for key, val in freq_dict.items() }

def build_distro(tokens):
    distro = {
        'uni': { },
        'bi': { }
     }
    '''
        uni: str as key
        bi: tuple(str, str) as key
    '''
    word_count = len(tokens)
    for i, word in enumerate(tokens):
        w_in_dict = distro['uni'].get(word) # None or not
        distro['uni'][word] = distro['uni'][word] + 1 if w_in_dict else 1

        if i < word_count - 1:
            word2 = tokens[i + 1]
            bi = (word, word2)
            bigram_in_dict = distro['bi'].get(bi)
            distro['bi'][bi] = distro['bi'][bi] + 1 if bigram_in_dict else 1

    distro['uni'] = convert_to_probs(distro['uni'])
    distro['bi'] = convert_to_probs(distro['bi'])

    return distro
