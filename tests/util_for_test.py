import os

HERE = os.path.abspath(os.path.dirname(__file__))

def getTFLiteModel(key):
  model_dir = os.path.abspath(os.path.join(HERE, '../assets/models'))
  return os.path.abspath(os.path.join(model_dir, key + '.tflite'))

