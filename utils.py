from tensorflow.python.keras.models import model_from_json
import json
import os
import io

def load_model(exp_dir):
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