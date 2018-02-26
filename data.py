import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation
from tensorflow.python.keras.layers import LSTM, Dropout, Embedding
from tensorflow.python.keras.optimizers import Adam
from tokenizer import Tokenizer
import numpy as np
import io
import os
import random

GLOVE = os.path.join('/Users/kevin/data', 'glove.6B', 'glove.6B.100d.txt')
EMBED_DIM = 100

def rand_sequence(sequence, length):
  start_index = random.randint(0, len(sequence) - length - 1)
  sentence = list(sequence[start_index:start_index + length])
  return sentence

def sequence_to_text(sequence, idx_word):
  string = ''
  for i in sequence:
    if i < len(idx_word):
      string += idx_word[i]
    else:
      string += 'out-of-vocab'

    string += ' '
  return string

def get_embeddings():
  print('Indexing word embeddings…')
  embeddings_idx = {}
  with io.open(GLOVE, 'r') as f:
    for line in f:
      values = line.split()
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embeddings_idx[word] = coefs

  return embeddings_idx

def get_data():
  print('Loading text dataset…')
  files = ['data/up1.txt', 'data/up2.txt', 'data/mr1.txt']
  text = ''
  for file in files:
    with io.open(file, encoding='utf-8') as f:
      text += f.read()
      text += ' '

  return text

def get_indices_and_sequence(text, max_words):
  print('Tokenizing and processing text…')
  tokenizer = Tokenizer(max_words=max_words)
  idx_word, word_idx, sequence = tokenizer.tokenize(text)
  return idx_word, word_idx, sequence

def get_embedding_matrix(idx_word, embeddings_idx):
  print('Generating embed matrix')
  corp_size = len(idx_word)
  embedding_matrix = np.zeros((corp_size + 1, 100))
  for i, word in enumerate(idx_word):
    embedding = embeddings_idx.get(word)
    if embedding is not None:
      embedding_matrix[i] = embedding
    else:
      embedding_matrix[i] = np.random.rand(EMBED_DIM)

  embedding_matrix[corp_size] = np.random.rand(EMBED_DIM)

  return embedding_matrix

def get_train_test_sets(sequence, sequence_length, idx_word):
  print('Generating train/test split')
  sentences = []
  next_words = []
  step = 3
  corp_size = len(idx_word)
  data_size = len(sequence)

  for i in range(0, data_size - sequence_length, step):
    if sequence[i+sequence_length] < corp_size:
      sentences.append(sequence[i:i+sequence_length])
      next_words.append(sequence[i+sequence_length])

  num_features = len(sentences)
  x = np.array(sentences)
  y = np.zeros((num_features, corp_size))

  for i in range(num_features):
    y[i][next_words[i]] = 1

  test_split = 0.15
  split_point = int(test_split * num_features)

  x_test = x[:split_point]
  y_test = y[:split_point]
  x_train = x[split_point:]
  y_train = y[split_point:]

  return x_train, y_train, x_test, y_test
