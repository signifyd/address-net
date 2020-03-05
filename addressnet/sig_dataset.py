from typing import Optional, Union, Callable, List
from collections import OrderedDict

import random
import tensorflow as tf
import numpy as np
import string

# Schema used to decode data from the TFRecord file
_features = OrderedDict([
    ('building_name', tf.io.FixedLenFeature([], tf.string)),
    ('lot_number_prefix', tf.io.FixedLenFeature([], tf.string)),
    ('lot_number', tf.io.FixedLenFeature([], tf.string)),
    ('lot_number_suffix', tf.io.FixedLenFeature([], tf.string)),
    ('flat_number_prefix', tf.io.FixedLenFeature([], tf.string)),
    ('flat_number_suffix', tf.io.FixedLenFeature([], tf.string)),
    ('level_number_prefix', tf.io.FixedLenFeature([], tf.string)),
    ('level_number_suffix', tf.io.FixedLenFeature([], tf.string)),
    ('number_first_prefix', tf.io.FixedLenFeature([], tf.string)),
    ('number_first_suffix', tf.io.FixedLenFeature([], tf.string)),
    ('number_last_prefix', tf.io.FixedLenFeature([], tf.string)),
    ('number_last_suffix', tf.io.FixedLenFeature([], tf.string)),
    ('street_name', tf.io.FixedLenFeature([], tf.string)),
    ('locality_name', tf.io.FixedLenFeature([], tf.string)),
    ('postcode', tf.io.FixedLenFeature([], tf.string)),
    ('flat_number', tf.io.FixedLenFeature([], tf.int64)),
    ('level_number', tf.io.FixedLenFeature([], tf.int64)),
    ('number_first', tf.io.FixedLenFeature([], tf.int64)),
    ('number_last', tf.io.FixedLenFeature([], tf.int64)),
    ('flat_type', tf.io.FixedLenFeature([], tf.int64)),
    ('level_type', tf.io.FixedLenFeature([], tf.int64)),
    ('street_type_code', tf.io.FixedLenFeature([], tf.int64)),
    ('street_suffix_code', tf.io.FixedLenFeature([], tf.int64)),
    ('state_abbreviation', tf.io.FixedLenFeature([], tf.int64)),
    ('latitude', tf.io.FixedLenFeature([], tf.float32)),
    ('longitude', tf.io.FixedLenFeature([], tf.float32))
])

# List of fields used as labels in the training data
labels_list = [
    'street_number', #1
    'street_name', #2
    'unit' #3
]
# Number of labels in total (+1 for the blank category)
n_labels = len(labels_list) + 1

# Allowable characters for the encoded representation
vocab = list(string.digits + string.ascii_lowercase + string.punctuation + string.whitespace)

def vocab_lookup(characters: str) -> (np.ndarray):
    """
    Converts a string into a list of vocab indices
    :param characters: the string to convert
    :param training: if True, artificial typos will be introduced
    :return: the string length and an array of vocab indices
    """
    result = list()
    for c in characters.lower():
        try:
            result.append(vocab.index(c) + 1)
        except ValueError:
            result.append(0)
    return np.array(result, dtype=np.int64)

def convert_labels(encoded_addr: str) -> (np.ndarray):
    labels_matrix = np.zeros((len(encoded_addr), 4), dtype=np.bool)
    for idx, label in enumerate(encoded_addr):
        labels_matrix[idx, int(label)] = True
    return labels_matrix

def synthesise_address(*record) -> (int, np.ndarray, np.ndarray):
    raw_and_encoded = str(record[0].numpy(), 'utf-8')
    splited = raw_and_encoded.split('|')
    raw_address = splited[0]
    encoded_addr = splited[1]
    text_encoded = vocab_lookup(raw_address)
    labels = convert_labels(encoded_addr)
    return (len(raw_address), text_encoded, labels)

def dataset(filenames: [str], batch_size: int = 10, shuffle_buffer: int = 1000, prefetch_buffer_size: int = 10000,
            num_parallel_calls: int = 8) -> Callable:
    """
    Creates a Tensorflow dataset and iterator operations
    :param filenames: the tfrecord filenames
    :param batch_size: training batch size
    :param shuffle_buffer: shuffle buffer size
    :param prefetch_buffer_size: size of the prefetch buffer
    :param num_parallel_calls: number of parallel calls for the mapping functions
    :return: the input_fn
    """

    def input_fn() -> tf.data.Dataset:
        ds = tf.data.TextLineDataset(filenames)
        ds = ds.map(lambda record: tf.py_function(synthesise_address, [record], [tf.int64, tf.int64, tf.bool]),
                    num_parallel_calls=num_parallel_calls)

        ds = ds.padded_batch(batch_size, ([], [None], [None, n_labels]))

        ds = ds.map(
            lambda _lengths, _encoded_text, _labels: ({'lengths': _lengths, 'encoded_text': _encoded_text}, _labels),
            num_parallel_calls=num_parallel_calls
        )
        ds = ds.prefetch(buffer_size=prefetch_buffer_size)
        return ds

    return input_fn


def predict_input_fn(input_text: List[str]) -> Callable:
    """
    An input function for one prediction example
    :param input_text: the input text
    :return:
    """

    def input_fn() -> tf.data.Dataset:
        predict_ds = tf.data.Dataset.from_generator(
            lambda: (vocab_lookup(address) for address in input_text),
            (tf.int64, tf.int64),
            (tf.TensorShape([]), tf.TensorShape([None]))
        )
        predict_ds = predict_ds.batch(1)
        predict_ds = predict_ds.map(
            lambda lengths, encoded_text: {'lengths': lengths, 'encoded_text': encoded_text}
        )
        return predict_ds

    return input_fn

def main():
    input_fn = dataset(['/Users/xxx/tmp/tf/encoded_US_100000.txt'])
    ds = input_fn()
    for element in ds:
        print(element)

if __name__ == '__main__':
    main()