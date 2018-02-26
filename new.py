import os
import argparse
import io
import model_def
import json

def parse_args():
  parser = argparse.ArgumentParser(description='Get the user notes on the new experiment')
  parser.add_argument('--notes', type=str, default='')
  parser.add_argument('--max_words', type=int, default=10000)
  parser.add_argument('--sequence_length', type=int, default=20)
  parser.add_argument('--rnn_size', type=int, default=128)
  parser.add_argument('--rnn_layers', type=int, default=1)
  parser.add_argument('--epoch', type=int, default=0)
  parser.add_argument('--batch_size', type=int, default=50)
  return parser.parse_args()

def main():
  # Scaffold new experiment directory
  experiments = os.listdir('experiments')
  most_recent = 0
  args = parse_args()

  for d in experiments:
    most_recent = int(d) if int(d) > most_recent else most_recent

  exp_dir = os.path.join('experiments', str(most_recent + 1))
  os.mkdir(exp_dir)
  os.mkdir(os.path.join(exp_dir, 'tb_logs'))
  os.mkdir(os.path.join(exp_dir, 'checkpoints'))
  os.mkdir(os.path.join(exp_dir, 'generated'))

  # Create notes file with notes from cmd line
  with io.open(os.path.join(exp_dir, 'README.txt'), 'w') as f:
    f.write(args.notes)

  print('Scaffolded new experiment directory')

  # Create model
  model = model_def.get_model(args.max_words, args.sequence_length, args.rnn_size, args.rnn_layers)
  m_json = model.to_json()

  # Save model to directory
  with io.open(os.path.join(exp_dir, 'model.json'), 'w') as f:
    f.write(m_json)

  with io.open(os.path.join(exp_dir, 'config.json'), 'w') as f:
    json.dump(vars(args), f)

  model.save_weights(os.path.join(exp_dir, 'checkpoints', '_initial.h5'))

  print('Experiment ' + str(most_recent + 1) + 'created! Train with "train.py --exp ' + str(most_recent + 1) + '"' )

if __name__ == '__main__':
  main()