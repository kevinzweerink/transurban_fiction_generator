from collections import Counter
import numpy as np

class Tokenizer(object):

  def __init__(self, max_words=None):
    self.max_words = max_words if max_words else int('inf')
    self.filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    self.split_char = ' '
    self.filtermap = str.maketrans(self.filters, self.split_char * len(self.filters))

  def untuple_counter(self, tup):
    word, count = tup
    return word

  def tokenize(self, text):
    '''
    returns index->word array, word->index dict, integer sequence
    '''
    text = text.lower()
    text = text.translate(self.filtermap)
    seq = text.split(self.split_char)
    seq = [i for i in seq if i]
    idx = Counter(seq).most_common(self.max_words)

    idx_word = list(map(self.untuple_counter, idx))
    word_idx = {}
    int_seq = np.zeros(len(seq), dtype=object)

    for i, w in enumerate(idx_word):
      word_idx[w]=i

    for i, w in enumerate(seq):
      if w in word_idx:
        int_seq[i] = word_idx[w]
      else:
        # if the word is outside of the max_words, assign an integer 1 higher than the length of the index
        int_seq[i] = len(idx_word)

    return idx_word, word_idx, int_seq