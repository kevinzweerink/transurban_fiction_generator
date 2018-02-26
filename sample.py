import data
import argparse
import utils
from tensorflow.python.keras.optimizers import Adam
import numpy as np
import os
import json

def parse_args():
  parser = argparse.ArgumentParser(description='Get the details of what to train')
  parser.add_argument('--length', type=int, default=250)
  parser.add_argument('--punctuation', type=float, default=0.09)
  parser.add_argument('--exp', type=int, default=1)
  parser.add_argument('--seed', type=list, default=None)

  return parser.parse_args()

def generate_seed(sequence, config):
  return data.rand_sequence(sequence, config['sequence_length'])

def sample(model, seed=None, length=250):
  if seed == None:
    print('Can\'t sample without a seed')
    return

  generated = seed[:]

  for i in range(0,length):
    x = np.array([seed])
    preds = model.predict(x, verbose=0)[0]
    next_word = np.random.choice(range(0, len(preds)), p=preds)
    seed = seed[1:]
    seed.append(next_word)
    generated.append(next_word)

  return generated

def punctuate(sequence, punctuation_rate):
  sequence = sequence.split()
  punctuations = ['.',',','!','?']
  probs = [0.6,0.2,0.1,0.1]
  for i, word in enumerate(sequence):
    if np.random.random() < punctuation_rate:
      sequence[i] += np.random.choice(punctuations, p=probs)


  return ' '.join(sequence)

def get_config(exp_dir):
  with open(os.path.join(exp_dir, 'config.json'), 'r') as f:
    config = json.loads(f.read())
  return config

def main():
  args = parse_args()
  exp_dir = os.path.join('experiments', str(args.exp))
  config = get_config(exp_dir)
  text = data.get_data()
  idx_word, _, sequence = data.get_indices_and_sequence(text, config['max_words'])
  seed = args.seed if args.seed else generate_seed(sequence, config)
  model = utils.load_model(exp_dir)
  model.compile(loss='categorical_crossentropy', optimizer=Adam())
  g = sample(model, length=args.length, seed=seed)
  g = punctuate(data.sequence_to_text(g, idx_word), args.punctuation)
  print(g)

if __name__ == '__main__':
  main()