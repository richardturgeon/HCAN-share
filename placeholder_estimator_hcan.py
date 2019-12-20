import glob
import json
import logging
import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import layer_norm
from tensorflow.python.eager.context import eager_mode, graph_mode

#tf.enable_eager_execution()

MYPATH = os.getcwd()

#BATCH_SIZE     = 32             # WAG
BATCH_SIZE     = 1
SHUFFLE_BUFFER = 1500           # from DECAN

#https://stackoverflow.com/questions/52266000/avoiding-tf-data-dataset-from-tensor-slices-with-estimator-api

class IteratorInitializerHook(tf.train.SessionRunHook):
    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        self.iterator_initializer_func(session)

def make_input_fn(X, y, shuffle=None, batch_size=1):
    iterator_initializer_hook = IteratorInitializerHook()

    def input_fn():
        X_pl = tf.placeholder(X.dtype, X.shape)
        y_pl = tf.placeholder(y.dtype, y.shape)
        dataset = tf.data.Dataset.from_tensor_slices((X_pl, y_pl))

        if shuffle:
            dataset = dataset.shuffle(1000)

        dataset = dataset.batch(batch_size)
        iterator = dataset.make_initializable_iterator()
        next_features, next_label = iterator.get_next()

        init_fn = lambda sess: sess.run(iterator.initializer, feed_dict={X_pl:X, y_pl: y})
        iterator_initializer_hook.iterator_initializer_func=init_fn
        return next_features, next_label

    return input_fn, iterator_initializer_hook


def logger(prefix):
    """
    Logger abstraction

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

##_____________________________________________________________________________

example_format = {
    'data':   tf.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True, default_value=0),
    'breaks': tf.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True, default_value=0),
    'label':  tf.FixedLenFeature([5], dtype=tf.int64)
}

def parse_fn(example):
    parsed_record = tf.parse_single_example(example, example_format)
    features = parsed_record['data']
    breaks = parsed_record['breaks']
    labels = parsed_record['label']
    return (features, breaks), labels

def tfrecord_loader(data_dir, filenames, nrecords=None, params=None):
    raw_dataset = tf.data.TFRecordDataset(filenames)
    parsed_dataset = raw_dataset.map(parse_fn)

    max_words = params['max_review_words']
    max_sentences = params['max_review_sentences']
    max_sentence_words = params['max_sentence_words']

    feature_list = []
    label_list = []

    for count, ((raw_text, breaks), labels) in enumerate(parsed_dataset):
        if count % 100 == 0:
            sys.stdout.write(f"loading record {count} of {nrecords}             \r")
            sys.stdout.flush()

        # ????????????????????????????????????????????????????????????????????
        #if count == 10000:
        #    break
        # ????????????????????????????????????????????????????????????????????

        np_raw_text = np.array(raw_text)
        np_breaks = np.array(breaks)
        np_labels = np.array(labels)
        sentence_origin = 0

        sentences = np.zeros([max_sentences, max_sentence_words], dtype='int32')

        for sentence_nbr, sentence_break in enumerate(np_breaks):
            if sentence_nbr >= max_sentences:
                break
            sentence_len = sentence_break - sentence_origin
            sentence_end = sentence_origin + sentence_len
            sentences[sentence_nbr,:sentence_len] = np_raw_text[sentence_origin:sentence_end]
            sentence_origin = sentence_break

        feature_list.append(sentences)
        label_list.append(np_labels)

    np_features = np.array(feature_list)
    np_labels = np.array(label_list)
    return np_features, np_labels

def main(train=False, eval=False, data_dir=None, model_dir=None, epochs=None, inpfx=''):
    """ """

    file_prefix = inpfx
    if file_prefix:
        file_prefix = file_prefix + '-'

    # load pre-processed data
    logger(file_prefix)
    print("loading data")
    vocab = np.load('data/yelp16_embeddings.npy')

    with open(file_prefix + 'HCAN-metadata.json', 'r') as f:
        metadata = json.load(f)

    params = {}
    params['max_review_words'] = metadata['max_review_words']
    params['max_review_sentences'] = metadata['max_review_sentences']
    params['max_sentence_words'] = metadata['max_sentence_words']
    params['nbr_classes'] = metadata['classes']

    train_set_size = metadata['train_count']
    test_set_size = metadata ['test_count']

    if params['max_review_sentences'] > 10:                                      # a practical limit ?????????????????????????
        params['max_review_sentences'] = 10

    params['model_dir'] = model_dir
    params['data_dir'] = data_dir
    params['vocab'] = vocab
    params['attention_heads'] = 8
    params['attention_size'] = 512
    params['dropout_keep'] = 0.9
    params['activation'] = tf.nn.elu
    params['file_prefix'] = file_prefix
    params['activation'] = tf.nn.elu

    train_steps = round(train_set_size / BATCH_SIZE) * epochs
    test_steps = round(test_set_size / BATCH_SIZE)

    print(f"Batch size is {BATCH_SIZE}")
    print(f"Scope as expressed in epochs {epochs}")
    print(f"Inputs - train set size: {train_set_size}, test set size: {test_set_size}")
    print(f"Train steps: {train_steps} test steps: {test_steps}")
    print(params)

    # build estimator ---------------------------------------------------------
    model = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params=params)

    # train model -------------------------------------------------------------
    if train:
        train_filename = get_filenames(data_dir, is_training=True, prefix=file_prefix)

        print("\nLoading training data...")
        with eager_mode():
            reviews, labels = tfrecord_loader(
                data_dir,
                [train_filename],
                nrecords=train_set_size,
                params=params)

        print("\nTraining...")
        train_input_fn, train_iter_init_hook = make_input_fn(reviews, labels, shuffle=True, batch_size=BATCH_SIZE)

        if tf.executing_eagerly():
            print("... training: taking from input_fn...")
            for feature_batch, label_batch in train_input_fn().take(1):                             # debug ??????????
                print(feature_batch)                                                                # debug ??????????
                print(label_batch)                                                                  # debug ??????????

        model.train(
            input_fn=train_input_fn,
            hooks=[train_iter_init_hook],
            steps=train_steps)
        print("Training complete!")

    # evaluate model ----------------------------------------------------------
    if eval:
        valid_filename = get_filenames(data_dir, is_training=False, prefix=file_prefix)

        print("\nLoading test data...")
        with eager_mode():
            reviews, labels = tfrecord_loader(
                data_dir,
                [valid_filename],
                nrecords=test_set_size,
                params=params)

        print("\nEvaluating...")
        eval_input_fn, eval_iter_init_hook = make_input_fn(reviews, labels, shuffle=False, batch_size=BATCH_SIZE)

        if tf.executing_eagerly():
            print("... evaluating: taking from input_fn...")
            for feature_batch, label_batch in eval_input_fn().take(1):                              # debug ??????????
                print(feature_batch)                                                                # debug ??????????
                print(label_batch)                                                                  # debug ??????????

        print("Evaluating...")
        eval_result = model.evaluate(
            input_fn=eval_input_fn,
            hooks=[eval_iter_init_hook],
            steps=train_steps)
        print("Evaluation complete! \n")

        print("global step:%7d" % eval_result['global_step'])
        print("accuracy:   %7.2f" % round(eval_result['accuracy'] * 100.0, 2))
        print("loss:       %7.2f" % round(eval_result['loss'], 2))

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
def model_fn(features, labels, mode, params):
    """ """
    nbr_classes = params['nbr_classes']
    p_max_words = params['max_review_words']
    p_max_sentences = params['max_review_sentences']            # self.ms in reference             
    p_max_sentence_words = params['max_sentence_words']         # self.mw in reference    
    attention_heads = params['attention_heads']
    attention_size = params['attention_size']
    embedding_matrix = params['vocab']
    dropout_keep = params['dropout_keep']
    activation = params['activation']

    embedding_size = embedding_matrix.shape[1]
    embeddings = embedding_matrix.astype(np.float32)

    """
    if mode == tf.estimator.ModeKeys.TRAIN:     # ???????? not yet enabled ???????
        use_dropout = True
    else:
        use_dropout = False
    """

    #doc input and mask
#   labels = labels[0]
#   labels = tf.reshape(labels, [nbr_classes])
#   doc_input = features['x']                                     # of dimension (BATCH_SZ, p_max_sentences, p_max_sentence_words)
#   doc_debatched = doc_input[0]

    doc_input = features[0]                                            # of dimension (p_max_sentences, p_max_sentence_words)

    words_per_line = tf.reduce_sum(tf.sign(doc_input), 1)
    max_lines = tf.reduce_sum(tf.sign(words_per_line))
    max_words = tf.reduce_max(words_per_line)
    doc_input_reduced = doc_input[:max_lines, :max_words]
    nbr_words = words_per_line[:max_lines]

    #word embeddings
    positions = tf.expand_dims(tf.range(max_words), 0)
    word_positions =  tf.gather(
        tf.get_variable(
            'word_pos',
            shape=(p_max_words, embedding_size),
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(0, 0.1)
        ),
        positions
    )
    word_embeds = tf.gather(
        tf.get_variable('embeddings', initializer=embeddings, dtype=tf.float32),
        doc_input_reduced
    )
    word_embeds = tf.nn.dropout(word_embeds + word_positions, dropout_keep)

    #masks to eliminate padding - ref lines 73-76
    mask_base = tf.cast(tf.sequence_mask(nbr_words, max_words), tf.float32)
    mask  = tf.tile(tf.expand_dims(mask_base, 2), [1, 1, attention_size])
    mask2 = tf.tile(tf.expand_dims(mask_base, 2), [attention_heads, 1, max_words])
    print(mask_base)
    print(mask)
    print(mask2)

    #word attention 1
    Q1 = tf.layers.conv1d(word_embeds, attention_size, 3, padding='same',
        activation=activation, kernel_initializer=tf.orthogonal_initializer())
    K1 = tf.layers.conv1d(word_embeds, attention_size, 3, padding='same',
        activation=activation, kernel_initializer=tf.orthogonal_initializer())
    V1 = tf.layers.conv1d(word_embeds, attention_size, 3, padding='same',
        activation=activation, kernel_initializer=tf.orthogonal_initializer())

    Q1 = tf.where(tf.equal(mask, 0), tf.zeros_like(Q1), Q1)
    K1 = tf.where(tf.equal(mask, 0), tf.zeros_like(K1), K1)
    V1 = tf.where(tf.equal(mask, 0), tf.zeros_like(V1), V1)

    Q1_ = tf.concat(tf.split(Q1, attention_heads, axis=2), axis=0)
    K1_ = tf.concat(tf.split(K1, attention_heads, axis=2), axis=0)
    V1_ = tf.concat(tf.split(V1, attention_heads, axis=2), axis=0)

    outputs1 = tf.matmul(Q1_, tf.transpose(K1_, [0, 2, 1]))
    outputs1 = outputs1/(K1_.get_shape().as_list()[-1] ** 0.5)
    outputs1 = tf.where(tf.equal(outputs1, 0), tf.ones_like(outputs1) * -1000, outputs1)
    outputs1 = tf.nn.dropout(tf.nn.softmax(outputs1), dropout_keep)
    outputs1 = tf.where(tf.equal(mask2, 0),tf.zeros_like(outputs1), outputs1)
    outputs1 = tf.matmul(outputs1, V1_)
    outputs1 = tf.concat(tf.split(outputs1, attention_heads, axis=0), axis=2)
    outputs1 = tf.where(tf.equal(mask, 0), tf.zeros_like(outputs1), outputs1)

    #word  attention 2
    Q2 = tf.layers.conv1d(word_embeds, attention_size, 3, padding='same',
        activation=activation, kernel_initializer=tf.orthogonal_initializer())
    K2 = tf.layers.conv1d(word_embeds, attention_size, 3, padding='same',
        activation=activation, kernel_initializer=tf.orthogonal_initializer())
    V2 = tf.layers.conv1d(word_embeds, attention_size, 3, padding='same',
        activation=tf.nn.tanh, kernel_initializer=tf.orthogonal_initializer())

    Q2 = tf.where(tf.equal(mask, 0), tf.zeros_like(Q2), Q2)
    K2 = tf.where(tf.equal(mask, 0), tf.zeros_like(K2), K2)
    V2 = tf.where(tf.equal(mask, 0), tf.zeros_like(V2), V2)

    Q2_ = tf.concat(tf.split(Q2, attention_heads, axis=2), axis=0)
    K2_ = tf.concat(tf.split(K2, attention_heads, axis=2), axis=0)
    V2_ = tf.concat(tf.split(V2, attention_heads, axis=2), axis=0)

    outputs2 = tf.matmul(Q2_, tf.transpose(K2_, [0, 2, 1]))
    outputs2 = outputs2/(K2_.get_shape().as_list()[-1] ** 0.5)
    outputs2 = tf.where(tf.equal(outputs2, 0), tf.ones_like(outputs2) * -1000, outputs2)
    outputs2 = tf.nn.dropout(tf.nn.softmax(outputs2), dropout_keep)
    outputs2 = tf.where(tf.equal(mask2, 0), tf.zeros_like(outputs2), outputs2)
    outputs2 = tf.matmul(outputs2, V2_)
    outputs2 = tf.concat(tf.split(outputs2, attention_heads, axis=0), axis=2)
    outputs2 = tf.where(tf.equal(mask, 0), tf.zeros_like(outputs2), outputs2)

    outputs = tf.multiply(outputs1, outputs2)
    outputs = layer_norm(outputs)

    #word target attention
    Q = tf.get_variable('word_Q', (1, 1, attention_size),
        tf.float32, tf.orthogonal_initializer())
    K = tf.layers.conv1d(outputs, attention_size, 3, padding='same',
        activation=activation, kernel_initializer=tf.orthogonal_initializer())

    Q = tf.tile(Q,[max_lines, 1, 1])
    K = tf.where(tf.equal(mask, 0),tf.zeros_like(K), K)

    Q_ = tf.concat(tf.split(Q, attention_heads, axis=2), axis=0)
    K_ = tf.concat(tf.split(K, attention_heads, axis=2), axis=0)
    V_ = tf.concat(tf.split(outputs, attention_heads, axis=2), axis=0)

    outputs = tf.matmul(Q_,tf.transpose(K_,[0, 2, 1]))
    outputs = outputs/(K_.get_shape().as_list()[-1]**0.5)
    outputs = tf.where(tf.equal(outputs,0),tf.ones_like(outputs)*-1000,outputs)
    outputs = tf.nn.dropout(tf.nn.softmax(outputs), dropout_keep)
    outputs = tf.matmul(outputs,V_)
    outputs = tf.concat(tf.split(outputs, attention_heads, axis=0), axis=2)
    sent_embeds = tf.transpose(outputs,[1, 0, 2])

    #sentence positional embeddings
    positions = tf.expand_dims(tf.range(max_lines),0)
    sent_pos = tf.gather(
        tf.get_variable('sent_pos',shape=(p_max_sentences, attention_size),
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(0,0.1)),
        positions)
    sent_embeds = tf.nn.dropout(sent_embeds + sent_pos, dropout_keep)

    #sentence attention 1
    Q1 = tf.layers.conv1d(sent_embeds, attention_size, 3, padding='same',
        activation=activation, kernel_initializer=tf.orthogonal_initializer())
    K1 = tf.layers.conv1d(sent_embeds, attention_size, 3, padding='same',
        activation=activation, kernel_initializer=tf.orthogonal_initializer())
    V1 = tf.layers.conv1d(sent_embeds, attention_size, 3, padding='same',
        activation=activation, kernel_initializer=tf.orthogonal_initializer())

    Q1_ = tf.concat(tf.split(Q1, attention_heads, axis=2), axis=0)
    K1_ = tf.concat(tf.split(K1, attention_heads, axis=2), axis=0)
    V1_ = tf.concat(tf.split(V1, attention_heads, axis=2), axis=0)

    outputs1 = tf.matmul(Q1_,tf.transpose(K1_,[0, 2, 1]))
    outputs1 = outputs1/(K1_.get_shape().as_list()[-1]**0.5)
    outputs1 = tf.nn.dropout(tf.nn.softmax(outputs1), dropout_keep)
    outputs1 = tf.matmul(outputs1,V1_)
    outputs1 = tf.concat(tf.split(outputs1, attention_heads, axis=0), axis=2)

    #sentence attention 2
    Q2 = tf.layers.conv1d(sent_embeds, attention_size, 3, padding='same',
        activation=activation, kernel_initializer=tf.orthogonal_initializer())
    K2 = tf.layers.conv1d(sent_embeds, attention_size, 3, padding='same',
        activation=activation, kernel_initializer=tf.orthogonal_initializer())
    V2 = tf.layers.conv1d(sent_embeds, attention_size, 3, padding='same',
        activation=tf.nn.tanh, kernel_initializer=tf.orthogonal_initializer())

    Q2_ = tf.concat(tf.split(Q2, attention_heads, axis=2), axis=0)
    K2_ = tf.concat(tf.split(K2, attention_heads, axis=2), axis=0)
    V2_ = tf.concat(tf.split(V2, attention_heads, axis=2), axis=0)

    outputs2 = tf.matmul(Q2_,tf.transpose(K2_,[0, 2, 1]))
    outputs2 = outputs2/(K2_.get_shape().as_list()[-1]**0.5)
    outputs2 = tf.nn.dropout(tf.nn.softmax(outputs2), dropout_keep)
    outputs2 = tf.matmul(outputs2,V2_)
    outputs2 = tf.concat(tf.split(outputs2, attention_heads, axis=0), axis=2)

    outputs = tf.multiply(outputs1,outputs2)
    outputs = layer_norm(outputs)

    #sentence target attention
    Q = tf.get_variable('sent_Q',(1,1, attention_size),
        tf.float32,tf.orthogonal_initializer())
    K = tf.layers.conv1d(outputs, attention_size, 3, padding='same',
        activation=activation, kernel_initializer=tf.orthogonal_initializer())

    Q_ = tf.concat(tf.split(Q, attention_heads, axis=2), axis=0)
    K_ = tf.concat(tf.split(K, attention_heads, axis=2), axis=0)
    V_ = tf.concat(tf.split(outputs, attention_heads, axis=2), axis=0)

    outputs = tf.matmul(Q_,tf.transpose(K_,[0, 2, 1]))
    outputs = outputs/(K_.get_shape().as_list()[-1] ** 0.5)
    outputs = tf.nn.dropout(tf.nn.softmax(outputs), dropout_keep)
    outputs = tf.matmul(outputs, V_)
    outputs = tf.concat(tf.split(outputs, attention_heads, axis=0), axis=2)
    doc_embed = tf.nn.dropout(tf.squeeze(outputs, [0]), dropout_keep)

    #classification functions
    logits = tf.layers.dense(doc_embed, nbr_classes, kernel_initializer=tf.orthogonal_initializer())

    #compute predictions, predictions required for PREDICT mode 
    predicted_classes = tf.argmax(logits, axis=1)
#   prediction_is_not_used = tf.nn.softmax(logits)

    #compute loss, loss required for TRAIN and EVAL 
#   labels_rs = tf.expand_dims(labels, 0)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))            # 12/02/19
#   loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
#   loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels_rs))

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
    optimizer = tf.train.AdamOptimizer(2e-5, 0.9, 0.99)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(
        mode,
        predictions=predicted_classes,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics,
        training_hooks=[logging_hook]
    )

##______________________________________________________________________________
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # candle-like                    
    parser.add_argument('--model_dir',
                        default=os.path.join(MYPATH, 'hcan_model_dir'),
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

