import tensorflow as tf
from addressnet.model import model_fn
from addressnet.sig_dataset import dataset as dataset

params={'rnn_size':128, 'rnn_layers': 3}


nnetwork = tf.estimator.Estimator(
    model_fn=model_fn, model_dir='models', config=None, params=params, warm_start_from=None
)

nnetwork.train(
    input_fn=dataset(filenames='/Users/nancyru/Code/signifyd_addressnet/data/encoded_US_100000.txt'), hooks=None, steps=100, max_steps=None, saving_listeners=None
)

#train_spec = tf.estimator.TrainSpec(input_fn=lambda: model.input_fn(args.training_file_pattern, True),
 #                                       max_steps=config["total_steps"])
#tf.estimator.train_and_evaluate(nnetwork, train_spec, eval_spec)