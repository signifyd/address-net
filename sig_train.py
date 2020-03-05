from addressnet.model import model_fn as model_fn
from addressnet.sig_dataset import dataset as dataset

import tensorflow as tf

def main():
    model_dir = '/Users/niaschmald/tmp/tfmodel'
    input_fn = dataset(['/Users/niaschmald/tmp/tf/encoded_US_100000.txt'])
    # input_fn = dataset([])

    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir)
    estimator.train(input_fn, max_steps=10)

if __name__ == "__main__":
    main()
