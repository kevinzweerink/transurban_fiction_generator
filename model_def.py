import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation
from tensorflow.python.keras.layers import LSTM, Embedding
from tensorflow.python.keras.optimizers import Adam
from tokenizer import Tokenizer
import data

EMBED_DIM = 100

def get_model(max_words, sequence_length, rnn_size, rnn_layers):
  print('Gathering data to scaffold model…')
  embeddings = data.get_embeddings()
  text = data.get_data()
  idx_word, word_idx, sequence = data.get_indices_and_sequence(text, max_words)
  embedding_matrix = data.get_embedding_matrix(idx_word, embeddings)
  corp_size = len(idx_word)
  embedding_layer = Embedding(corp_size + 1, EMBED_DIM, weights=[embedding_matrix], input_length=sequence_length, trainable=False)

  model = Sequential()
  model.add(embedding_layer)

  for i in range(0, rnn_layers):
    output_seqs = True if i < rnn_layers - 1 else False
    model.add(LSTM(rnn_size, input_shape=(sequence_length, EMBED_DIM), return_sequences=output_seqs))

  model.add(Dense(corp_size))
  model.add(Activation('softmax'))
  model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01))

  return model


