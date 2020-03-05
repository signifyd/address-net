from addressnet.model import model_fn as model_fn
from addressnet.sig_dataset import dataset as dataset

import tensorflow as tf

def main():
    params={'rnn_size':128, 'rnn_layers': 3}

    model_dir = '/Users/niaschmald/tmp/tfmodel'
    nnetwork = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=model_dir, config=None, params=params, warm_start_from=None
    )

    input_fn = dataset(['/Users/niaschmald/tmp/tf/encoded_US_100000.txt'])

    nnetwork.train(
        input_fn=input_fn, hooks=None, steps=100, max_steps=None, saving_listeners=None
    )

if __name__ == "__main__":
    main()
