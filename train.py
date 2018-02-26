import data
import tensorflow as tf
from tensorflow.python.keras.callbacks import LambdaCallback, TensorBoard, ModelCheckpoint
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.optimizers import Adam
import json
import io
import argparse
import os
import time
import numpy as np

config = {}
sequence = []
word_idx = {}
idx_word = []
exp_dir = ''
model = None
args = None

def parse_args():
  parser = argparse.ArgumentParser(description='Get the details of what to train')
  parser.add_argument('--exp', type=int, default=1)
  parser.add_argument('--epochs', type=int, default=50)
  return parser.parse_args()

def get_config():
  global config
  with open(os.path.join(exp_dir, 'config.json'), 'r') as f:
    config = json.loads(f.read())
  return config

def save_config():
  with io.open(os.path.join(exp_dir, 'config.json'), 'w') as f:
    json.dump(config, f)

def load_model():
  with io.open(os.path.join(exp_dir, 'model.json'), 'r') as f:
    model = model_from_json(f.read())

  checkpoints_dir = os.path.join(exp_dir, 'checkpoints')
  weights = os.listdir(checkpoints_dir)
  if (len(weights) == 1 and '_initial.h5' in weights):
    model.load_weights(os.path.join(checkpoints_dir, '_initial.h5'))
  elif (len(weights) > 1):
    l = sorted(weights)
    best = max(l[:-1])
    model.load_weights(os.path.join(checkpoints_dir, best))

  return model

def save_generated(sequence):
  filename = 'E' + format(config['epoch'], '03') + '---' + time.strftime('%Y%m%d%H%M%S') + '.txt'
  with io.open(os.path.join(exp_dir, 'generated', filename), 'w') as f:
    f.write(data.sequence_to_text(sequence, idx_word))

def sample_test(epoch, logs):
  print('------- Finished training Epoch %d' % config['epoch'])
  sample_seed = data.rand_sequence(sequence, config['sequence_length'])
  generated = sample_seed[:]

  for i in range(250):
    x_pred = np.array([sample_seed])
    preds = model.predict(x_pred, verbose=0)[0]
    next_word = np.random.choice(range(0, len(preds)), p=preds)
    sample_seed = sample_seed[1:]
    sample_seed.append(next_word)
    generated.append(next_word)

  save_generated(generated)
  config['epoch'] += 1
  save_config()

def train(model, x, y, args):
  sample_callback = LambdaCallback(on_epoch_end=sample_test)
  tb_callback = TensorBoard(log_dir=os.path.join(exp_dir, 'tb_logs'), histogram_freq=0, write_graph=True, write_images=False)
  checkpoint = ModelCheckpoint(filepath=os.path.join(exp_dir, 'checkpoints', '{val_loss:.3f}--E{epoch:03d}.hdf5'))
  model.compile(loss='categorical_crossentropy', optimizer=Adam())
  model.fit(x, y, batchsize=config['batch_size'], epochs=args.epochs, callbacks=[sample_callback, tb_callback, checkpoint], validation_split=0.2)

def load_globals():
  global config, model, text, idx_word, word_idx, sequence, exp_dir, args
  args = parse_args()
  exp_dir = os.path.join('experiments', str(args.exp))

  if not os.path.exists(exp_dir):
    print('Could not locate experiment for training')
    print(exp_dir)
    return

  config = get_config()
  model = load_model()
  text = data.get_data()
  idx_word, word_idx, sequence = data.get_indices_and_sequence(text, config['max_words'])

def main():  
  x_train, y_train, x_test, y_test = data.get_train_test_sets(sequence, config['sequence_length'], idx_word)
  train(model, x_train, y_train, args)

if __name__ == '__main__':
  load_globals()
  main()

