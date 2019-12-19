import glob
import json
import logging
import os
import pickle                       # TFRecords 
import sys
import time

import numpy as np
import tensorflow as tf

MYPATH = os.getcwd()

BATCH_SIZE     = 32             # WAG
SHUFFLE_BUFFER = 1500           # from DECAN

tf.enable_eager_execution()


def logger(prefix):
    """
    Logger abstraction

    taken from DECAN/src/decan_utils.py :: logger()

    Returns
    -------
    Logger object
    """
    logging.getLogger().setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)-2s - %(levelname)-2s - %(message)s', "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(prefix + 'HCAN-estimator.log')
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(logging.INFO)

    logging.getLogger().addHandler(fh)
    logging.getLogger().addHandler(ch)
    return logging.getLogger(__name__)


def main(train=False, eval=False, data_dir=None, model_dir=None, epochs=None, inpfx=''):
    """ """

    file_prefix = inpfx
    if file_prefix:
        file_prefix = file_prefix + '-'

    # load pre-processed data
    logger(file_prefix)
    print("loading data")
    vocab = np.load('data/yelp16_embeddings_tfr.npy')

    with open(file_prefix + 'HCAN-metadata.json', 'r') as f:
        metadata = json.load(f)

    params = {}
    params['max_review_words'] = metadata['max_review_words']
    params['nbr_classes'] = metadata['classes']

    train_set_size = metadata['train_count']
    test_set_size = metadata ['test_count']

    params['model_dir'] = model_dir
    params['data_dir'] = data_dir
    params['vocab'] = vocab
    params['nbr_filters'] = 100
    params['dropout_keep'] = 0.5
    params['file_prefix'] = file_prefix

    train_steps = round(train_set_size / BATCH_SIZE) * epochs
    test_steps = round(test_set_size / BATCH_SIZE)

    print(f"Batch size is {BATCH_SIZE}")
    print(f"Scope as expressed in epochs {epochs}")
    print(f"Inputs - train set size: {train_set_size}, test set size: {test_set_size}")
    print(f"Train steps: {train_steps} test steps: {test_steps}")

    # build estimator
    model = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params=params)

    # train model
    if train:
        _input_fn = lambda: input_fn(data_dir, BATCH_SIZE, is_training=True, params=params)
        print("Training...")

        for feature_batch, label_batch in _input_fn().take(1):                                      # debug ??????????
            print(feature_batch)                                                                    # debug ??????????
            print(label_batch)                                                                      # debug ??????????

        model.train(input_fn=_input_fn, steps=train_steps)
        print("Trained!")

    # evaluate model
    if eval:
        print("Evaluating...")
        _eval_input_fn = lambda: input_fn(data_dir, BATCH_SIZE, is_training=False, params=params)

        for feature_batch, label_batch in _eval_input_fn().take(1):                                 # debug ??????????
            print(feature_batch)                                                                    # debug ??????????
            print(label_batch)                                                                      # debug ??????????

        eval_result = model.evaluate(input_fn=_eval_input_fn, steps=test_steps)
        print("global step:%7d" % eval_result['global_step'])
        print("accuracy:   %7.2f" % round(eval_result['accuracy'] * 100.0, 2))
        print("loss:       %7.2f" % round(eval_result['loss'], 2))
        print("Evaluation complete!")

# ------------------------------------------------------------------------------

def input_fn(data_dir, batch_size, is_training=None, prep_style='minimal', num_parallel_reads=0, params=None):

    feature_map = {
        'data':     tf.FixedLenSequenceFeature([], allow_missing=True, dtype=tf.int64, default_value=0),
        'label':    tf.FixedLenFeature([5], dtype=tf.int64)
    }

    file_prefix = params['file_prefix']
    filenames = get_filenames(data_dir, is_training, fmt='tfrecords', prefix=file_prefix)
    dataset   = tf.data.Dataset.from_tensor_slices(filenames)

    if is_training:
        dataset = dataset.shuffle(buffer_size=len(filenames))

    if num_parallel_reads >= 1:
        dataset = dataset.flat_map(lambda x: tf.data.TFRecordDataset(x, num_parallel_reads=num_parallel_reads))
    else:
        dataset = dataset.flat_map(tf.data.TFRecordDataset)

    def parse_record_fn(raw_record, is_training):
        return parse_record(raw_record, is_training=is_training, feature_map=feature_map, prep_style=prep_style, params=params)

    return process_record_dataset(dataset, is_training, batch_size, SHUFFLE_BUFFER, parse_record_fn)

##_____________________________________________________________________________
def get_filenames(data_dir, is_training=True, fmt='tfrecords', prefix=''):
    """Return filenames for dataset."""
    if is_training:
        tfrecords = glob.glob(os.path.join(data_dir, prefix + 'train*.%s' % fmt))
    else:
        tfrecords = glob.glob(os.path.join(data_dir, prefix + 'test*.%s' % fmt))
    tfrecords.sort()
    return tfrecords


##_____________________________________________________________________________
def process_record_dataset(dataset, is_training, batch_size, shuffle_buffer,
                           parse_record_fn, num_epochs=1, num_parallel_batches=1):
    """Given a Dataset with raw records, return an iterator over the records.

    Args:
      dataset: A Dataset representing raw records
      is_training: A boolean denoting whether the input is for training.
      batch_size: The number of samples per batch.
      shuffle_buffer: The buffer size to use when shuffling records. A larger
        value results in better randomness, but smaller values reduce startup
        time and use less memory.
      parse_record_fn: A function that takes a raw record and returns the
        corresponding (image, label) pair.
      num_epochs: The number of epochs to repeat the dataset.

    Returns:
      Dataset of (image, label) pairs ready for iteration.
    """

    num_epochs = None

    # We prefetch a batch at a time, This can help smooth out the time taken to
    # load input files as we go through shuffling and processing.
    dataset = dataset.prefetch(buffer_size=batch_size)

    # Shuffle the records. Note that we shuffle before repeating to ensure
    # that the shuffling respects epoch boundaries.
    if True: #is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)

    # If we are training over multiple epochs before evaluating, repeat the
    # dataset for the appropriate number of epochs.
    dataset = dataset.repeat(num_epochs)

    # Parse the raw records into images and labels. Testing has shown that setting
    # num_parallel_batches > 1 produces no improvement in throughput, since
    # batch_size is almost always much greater than the number of CPU cores.

    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
            lambda value: parse_record_fn(value, is_training=is_training),
            batch_size=batch_size,
            num_parallel_batches=num_parallel_batches,
            drop_remainder=True))

    # Operations between the final prefetch and the get_next call to the iterator
    # will happen synchronously during run time. We prefetch here again to
    # background all of the above processing work and keep it out of the
    # critical training path. Setting buffer_size to tf.contrib.data.AUTOTUNE
    # allows DistributionStrategies to adjust how many batches to fetch based
    # on how many devices are present.
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

    return dataset


### global_counter = 0

##_____________________________________________________________________________
def parse_record(raw_record, feature_map, is_training=True, prep_style='minimal', params=None):
    prep_style = prep_style.lower()
    assert prep_style == 'minimal'
    max_words = params['max_review_words']

    record_features = tf.parse_single_example(raw_record, feature_map)
    data  = record_features['data']
    data  = tf.cast(data, tf.int32)

    paddings = [[0, max_words - tf.shape(data)[0]]]
    padded = tf.pad(data, paddings, 'CONSTANT')
    #remap = tf.reshape(data, [tf.shape(data)[0], max_words])
    #remap = tf.reshape(data, [max_words])
    label = record_features['label']
    label = tf.cast(label, tf.int32)

    features = {'x' : padded}

    """                             11/06/2019 - records appear correct
    global global_counter
    if global_counter < 10:
        global_counter += 1
        tf.print(features, output_stream=sys.stdout)
        tf.print(label, output_stream=sys.stdout)
    """

    return features, label

##_____________________________________________________________________________

def model_fn(features, labels, mode, params):
    """ """
    max_words = params['max_review_words']
    nbr_classes = params['nbr_classes']
    nbr_filters = params['nbr_filters']
    dropout_keep = params['dropout_keep']
    embedding_matrix = params['vocab']

    vocab = embedding_matrix
    embedding_size = embedding_matrix.shape[1]
    embeddings = embedding_matrix.astype(np.float32)

    #doc input and mask
    doc_input = features['x']

    if mode == tf.estimator.ModeKeys.TRAIN:
        use_dropout = True
    else:
        use_dropout = False

    #word embeddings
    word_embeds = tf.gather(
        tf.get_variable('embeddings', initializer=embeddings, dtype=tf.float32),
        doc_input
    )

    #word convolutions
    conv3 = tf.layers.conv1d(word_embeds, nbr_filters, 3, padding='same',
            activation=tf.nn.relu, kernel_initializer=tf.orthogonal_initializer())
    conv4 = tf.layers.conv1d(word_embeds, nbr_filters, 4, padding='same',
            activation=tf.nn.relu, kernel_initializer=tf.orthogonal_initializer())
    conv5 = tf.layers.conv1d(word_embeds, nbr_filters, 5, padding='same',
            activation=tf.nn.relu, kernel_initializer=tf.orthogonal_initializer())
    pool3 = tf.reduce_max(conv3, 1)
    pool4 = tf.reduce_max(conv4, 1)
    pool5 = tf.reduce_max(conv5, 1)

    #concatenate
    concat = tf.concat([pool3, pool4, pool5], 1)
    doc_embed = tf.nn.dropout(concat, dropout_keep)

    #classification functions
    logits = tf.layers.dense(doc_embed, nbr_classes, kernel_initializer=tf.orthogonal_initializer())

    #compute predictions, predictions required for PREDICT mode 
    predicted_classes = tf.argmax(logits, axis=1)
    prediction_is_not_used = tf.nn.softmax(logits)

    #compute loss, loss required for TRAIN and EVAL 
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))

    #compute evaluation metrics, metrics required for EVAL 
    accuracy = tf.metrics.accuracy(
        labels=tf.argmax(labels, axis=1),
        predictions=predicted_classes,
        name='accuracy'
    )

    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    tf.logging.set_verbosity(tf.logging.INFO)
    logging_hook = tf.train.LoggingTensorHook(
        {"loss": loss, "accuracy": accuracy[1]},
        every_n_secs = 15
    )

    #create training op required for TRAIN
    optimizer = tf.train.AdamOptimizer(0.00002, 0.9, 0.99)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(
        mode,
        predictions=predicted_classes,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics,
        training_hooks = [logging_hook]
    )

##______________________________________________________________________________
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # candle-like                    
    parser.add_argument('--model_dir',
                        default=os.path.join(MYPATH, 'cnn_model_dir'),
                        type=str,
                        help='tensorflow model_dir.')

    parser.add_argument('--data_dir',
                        default='./',
                        type=str,
                        help='tensorflow data_dir.')

    parser.add_argument('--train',
                        default=False,
                        action='store_true',
                        help='train the model.')

    parser.add_argument('--eval',
                        default=False,
                        action='store_true',
                        help='evaluate the model.')

    parser.add_argument('--epochs',
                        default=10,
                        type=int,
                        help='Epochs (converted to estimator steps.')

    parser.add_argument('--inpfx',
                        default='',
                        type=str,
                        help='prefix prepended to tfrecord, JSON and log files')

    args = vars(parser.parse_args())
    main(**args)

