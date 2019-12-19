import glob
import json
import logging
import os
import pickle
import numpy as np
import tensorflow as tf
import sys
import time

# import tensorflow.contrib.eager as tfe
from tensorflow.python.eager.context import eager_mode, graph_mode

from tensorflow.contrib.layers import layer_norm
from sklearn.model_selection import train_test_split

# tf.enable_eager_execution()
MYPATH = os.getcwd()

class hcan(object):

    def __init__(self,embedding_matrix,num_classes,max_sents,max_words,attention_heads=8,
                 attention_size=512,dropout_keep=0.9,activation=tf.nn.elu):
        '''
        hierarchical convolutional attention network for text classification

        parameters:
          - embedding_matrix: numpy array
            numpy array of word embeddings
            each row should represent a word embedding
            NOTE: the word index 0 is dropped, so the first row is ignored
          - num_classes: int
            number of output classes
          - max_sents: int
            maximum number of sentences per document
          - max_words: int
            maximum number of words per sentence
          - attention_heads: int (default: 8)
            number of attention heads to use in multihead attention
          - attention_size: int (default: 512)
            dimension size of output embeddings from attention
          - dropout_keep: float (default: 0.9)
            dropout keep rate for embeddings and attention softmax
          - activation: tensorflow activation function (default: tf.nn.elu)
            activation function to use for convolutional feature extraction

        methods:
          - train(,data,labels,validation_data,epochs=30,savebest=False,filepath=None)
            train network on given data
          - predict(data)
            return the one-hot-encoded predicted labels for given data
          - score(data,labels)
            return the accuracy of predicted labels on given data
          - save(filepath)
            save the model weights to a file
          - load(filepath)
            load model weights from a file
        '''

        self.attention_heads = attention_heads
        self.attention_size = attention_size
        self.embedding_size = embedding_matrix.shape[1]
        self.embeddings = embedding_matrix.astype(np.float32)
        self.ms = max_sents
        self.mw = max_words
        self.dropout_keep = dropout_keep
        self.dropout = tf.placeholder(tf.float32)

        #doc input and mask
        self.doc_input = tf.placeholder(tf.int32, shape=[max_sents,max_words])
        self.words_per_line = tf.reduce_sum(tf.sign(self.doc_input),1)
        self.max_lines = tf.reduce_sum(tf.sign(self.words_per_line))
        self.max_words = tf.reduce_max(self.words_per_line)
        self.doc_input_reduced = self.doc_input[:self.max_lines,:self.max_words]
        self.num_words = self.words_per_line[:self.max_lines]

        #word embeddings
        self.word_embeds = tf.gather(tf.get_variable('embeddings',initializer=self.embeddings,
                           dtype=tf.float32),self.doc_input_reduced)
        positions = tf.expand_dims(tf.range(self.max_words),0)
        word_pos = tf.gather(tf.get_variable('word_pos',shape=(self.mw,self.embedding_size),
                   dtype=tf.float32,initializer=tf.random_normal_initializer(0,0.1)),positions)
        self.word_embeds = tf.nn.dropout(self.word_embeds + word_pos,self.dropout)

        #for feature/parameter comparison
        print(f"attention heads: {attention_heads}")
        print(f"attention size: {attention_size}")
        print(f"self embedding size: {self.embedding_size}")
        print(f"self embeddings: {self.embeddings}")
        print(f"max sents (ms): {self.ms}")
        print(f"max words (mw): {self.mw}")
        print(f"dropout: {dropout_keep}")

        print(f"self doc_input: {self.doc_input}")
        print(f"self words_per_line: {self.words_per_line}")
        print(f"self max_lines {self.max_lines}")
        print(f"self max_words {self.max_words}")
        print(f"self doc_input_reduced: {self.doc_input_reduced}")
        print(f"self num_words: {self.num_words}")

        #masks to eliminate padding
        mask_base = tf.cast(tf.sequence_mask(self.num_words,self.max_words),tf.float32)
        mask = tf.tile(tf.expand_dims(mask_base,2),[1,1,self.attention_size])
        mask2 = tf.tile(tf.expand_dims(mask_base,2),[self.attention_heads,1,self.max_words])
        print(f"mask_base: {mask_base}")
        print(f"mask: {mask}")
        print(f"mask2: {mask2}")

        #word self attention 1
        Q1 = tf.layers.conv1d(self.word_embeds,self.attention_size,3,padding='same',
            activation=activation,kernel_initializer=tf.orthogonal_initializer())
        K1 = tf.layers.conv1d(self.word_embeds,self.attention_size,3,padding='same',
            activation=activation,kernel_initializer=tf.orthogonal_initializer())
        V1 = tf.layers.conv1d(self.word_embeds,self.attention_size,3,padding='same',
            activation=activation,kernel_initializer=tf.orthogonal_initializer())

        Q1 = tf.where(tf.equal(mask,0),tf.zeros_like(Q1),Q1)
        K1 = tf.where(tf.equal(mask,0),tf.zeros_like(K1),K1)
        V1 = tf.where(tf.equal(mask,0),tf.zeros_like(V1),V1)

        Q1_ = tf.concat(tf.split(Q1,self.attention_heads,axis=2),axis=0)
        K1_ = tf.concat(tf.split(K1,self.attention_heads,axis=2),axis=0)
        V1_ = tf.concat(tf.split(V1,self.attention_heads,axis=2),axis=0)

        outputs1 = tf.matmul(Q1_,tf.transpose(K1_,[0, 2, 1]))
        outputs1 = outputs1/(K1_.get_shape().as_list()[-1]**0.5)
        outputs1 = tf.where(tf.equal(outputs1,0),tf.ones_like(outputs1)*-1000,outputs1)
        outputs1 = tf.nn.dropout(tf.nn.softmax(outputs1),self.dropout)
        outputs1 = tf.where(tf.equal(mask2,0),tf.zeros_like(outputs1),outputs1)
        outputs1 = tf.matmul(outputs1,V1_)
        outputs1 = tf.concat(tf.split(outputs1,self.attention_heads,axis=0),axis=2)
        outputs1 = tf.where(tf.equal(mask,0),tf.zeros_like(outputs1),outputs1)

        #word self attention 2
        Q2 = tf.layers.conv1d(self.word_embeds,self.attention_size,3,padding='same',
            activation=activation,kernel_initializer=tf.orthogonal_initializer())
        K2 = tf.layers.conv1d(self.word_embeds,self.attention_size,3,padding='same',
            activation=activation,kernel_initializer=tf.orthogonal_initializer())
        V2 = tf.layers.conv1d(self.word_embeds,self.attention_size,3,padding='same',
            activation=tf.nn.tanh,kernel_initializer=tf.orthogonal_initializer())

        Q2 = tf.where(tf.equal(mask,0),tf.zeros_like(Q2),Q2)
        K2 = tf.where(tf.equal(mask,0),tf.zeros_like(K2),K2)
        V2 = tf.where(tf.equal(mask,0),tf.zeros_like(V2),V2)

        Q2_ = tf.concat(tf.split(Q2,self.attention_heads,axis=2),axis=0)
        K2_ = tf.concat(tf.split(K2,self.attention_heads,axis=2),axis=0)
        V2_ = tf.concat(tf.split(V2,self.attention_heads,axis=2),axis=0)

        outputs2 = tf.matmul(Q2_,tf.transpose(K2_,[0, 2, 1]))
        outputs2 = outputs2/(K2_.get_shape().as_list()[-1]**0.5)
        outputs2 = tf.where(tf.equal(outputs2,0),tf.ones_like(outputs2)*-1000,outputs2)
        outputs2 = tf.nn.dropout(tf.nn.softmax(outputs2),self.dropout)
        outputs2 = tf.where(tf.equal(mask2,0),tf.zeros_like(outputs2),outputs2)
        outputs2 = tf.matmul(outputs2,V2_)
        outputs2 = tf.concat(tf.split(outputs2,self.attention_heads,axis=0),axis=2)
        outputs2 = tf.where(tf.equal(mask,0),tf.zeros_like(outputs2),outputs2)

        outputs = tf.multiply(outputs1,outputs2)
        outputs = layer_norm(outputs)

        #word target attention
        Q = tf.get_variable('word_Q',(1,1,self.attention_size),
            tf.float32,tf.orthogonal_initializer())
        K = tf.layers.conv1d(outputs,self.attention_size,3,padding='same',
            activation=activation,kernel_initializer=tf.orthogonal_initializer())

        Q = tf.tile(Q,[self.max_lines,1,1])
        K = tf.where(tf.equal(mask,0),tf.zeros_like(K),K)

        Q_ = tf.concat(tf.split(Q,self.attention_heads,axis=2),axis=0)
        K_ = tf.concat(tf.split(K,self.attention_heads,axis=2),axis=0)
        V_ = tf.concat(tf.split(outputs,self.attention_heads,axis=2),axis=0)

        outputs = tf.matmul(Q_,tf.transpose(K_,[0, 2, 1]))
        outputs = outputs/(K_.get_shape().as_list()[-1]**0.5)
        outputs = tf.where(tf.equal(outputs,0),tf.ones_like(outputs)*-1000,outputs)
        outputs = tf.nn.dropout(tf.nn.softmax(outputs),self.dropout)
        outputs = tf.matmul(outputs,V_)
        outputs = tf.concat(tf.split(outputs,self.attention_heads,axis=0),axis=2)
        self.sent_embeds = tf.transpose(outputs,[1, 0, 2])

        #sentence positional embeddings
        positions = tf.expand_dims(tf.range(self.max_lines),0)
        sent_pos = tf.gather(tf.get_variable('sent_pos',shape=(self.ms,self.attention_size),
                   dtype=tf.float32,initializer=tf.random_normal_initializer(0,0.1)),positions)
        self.sent_embeds = tf.nn.dropout(self.sent_embeds + sent_pos,self.dropout)

        #sentence self attention 1
        Q1 = tf.layers.conv1d(self.sent_embeds,self.attention_size,3,padding='same',
            activation=activation,kernel_initializer=tf.orthogonal_initializer())
        K1 = tf.layers.conv1d(self.sent_embeds,self.attention_size,3,padding='same',
            activation=activation,kernel_initializer=tf.orthogonal_initializer())
        V1 = tf.layers.conv1d(self.sent_embeds,self.attention_size,3,padding='same',
            activation=activation,kernel_initializer=tf.orthogonal_initializer())

        Q1_ = tf.concat(tf.split(Q1,self.attention_heads,axis=2),axis=0)
        K1_ = tf.concat(tf.split(K1,self.attention_heads,axis=2),axis=0)
        V1_ = tf.concat(tf.split(V1,self.attention_heads,axis=2),axis=0)

        outputs1 = tf.matmul(Q1_,tf.transpose(K1_,[0, 2, 1]))
        outputs1 = outputs1/(K1_.get_shape().as_list()[-1]**0.5)
        outputs1 = tf.nn.dropout(tf.nn.softmax(outputs1),self.dropout)
        outputs1 = tf.matmul(outputs1,V1_)
        outputs1 = tf.concat(tf.split(outputs1,self.attention_heads,axis=0),axis=2)

        #sentence self attention 2
        Q2 = tf.layers.conv1d(self.sent_embeds,self.attention_size,3,padding='same',
            activation=activation,kernel_initializer=tf.orthogonal_initializer())
        K2 = tf.layers.conv1d(self.sent_embeds,self.attention_size,3,padding='same',
            activation=activation,kernel_initializer=tf.orthogonal_initializer())
        V2 = tf.layers.conv1d(self.sent_embeds,self.attention_size,3,padding='same',
            activation=tf.nn.tanh,kernel_initializer=tf.orthogonal_initializer())

        Q2_ = tf.concat(tf.split(Q2,self.attention_heads,axis=2),axis=0)
        K2_ = tf.concat(tf.split(K2,self.attention_heads,axis=2),axis=0)
        V2_ = tf.concat(tf.split(V2,self.attention_heads,axis=2),axis=0)

        outputs2 = tf.matmul(Q2_,tf.transpose(K2_,[0, 2, 1]))
        outputs2 = outputs2/(K2_.get_shape().as_list()[-1]**0.5)
        outputs2 = tf.nn.dropout(tf.nn.softmax(outputs2),self.dropout)
        outputs2 = tf.matmul(outputs2,V2_)
        outputs2 = tf.concat(tf.split(outputs2,self.attention_heads,axis=0),axis=2)

        outputs = tf.multiply(outputs1,outputs2)
        outputs = layer_norm(outputs)

        #sentence target attention
        Q = tf.get_variable('sent_Q',(1,1,self.attention_size),
            tf.float32,tf.orthogonal_initializer())
        K = tf.layers.conv1d(outputs,self.attention_size,3,padding='same',
            activation=activation,kernel_initializer=tf.orthogonal_initializer())

        Q_ = tf.concat(tf.split(Q,self.attention_heads,axis=2),axis=0)
        K_ = tf.concat(tf.split(K,self.attention_heads,axis=2),axis=0)
        V_ = tf.concat(tf.split(outputs,self.attention_heads,axis=2),axis=0)

        outputs = tf.matmul(Q_,tf.transpose(K_,[0, 2, 1]))
        outputs = outputs/(K_.get_shape().as_list()[-1]**0.5)
        outputs = tf.nn.dropout(tf.nn.softmax(outputs),self.dropout)
        outputs = tf.matmul(outputs,V_)
        outputs = tf.concat(tf.split(outputs,self.attention_heads,axis=0),axis=2)
        self.doc_embed = tf.nn.dropout(tf.squeeze(outputs,[0]),self.dropout)

        #classification functions
        self.output = tf.layers.dense(self.doc_embed,num_classes,
                      kernel_initializer=tf.orthogonal_initializer())
        self.prediction = tf.nn.softmax(self.output)

        #loss, accuracy, and training functions
        self.labels = tf.placeholder(tf.float32, shape=[num_classes])
        self.labels_rs = tf.expand_dims(self.labels,0)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                    (logits=self.output,labels=self.labels_rs))
        self.optimizer = tf.train.AdamOptimizer(2e-5,0.9,0.99).minimize(self.loss)

        #init op
        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(self.init_op)

#------------------------------------------------------------------------------
    def _list_to_numpy(self,inputval):
        '''
        convert variable length lists of input values to zero padded numpy array
        '''
        if type(inputval) == list:
            retval = np.zeros((self.ms,self.mw))
            for i,line in enumerate(inputval):
                for j, word in enumerate(line):
                    retval[i,j] = word
            return retval
        elif type(inputval) == np.ndarray:
            return inputval
        else:
            raise Exception("invalid input type")

    def train(self,data,labels,validation_data,epochs=5,savebest=False,filepath=None):
        '''
        train network on given data

        parameters:
          - data: numpy array
            3d numpy array (doc x sentence x word ids) of input data
          - labels: numpy array
            2d numpy array of one-hot-encoded labels
          - epochs: int (default: 30)
            number of epochs to train for
          - validation_data: tuple (optional)
            tuple of numpy arrays (X,y) representing validation data
          - savebest: boolean (default: False)
            set to True to save the best model based on validation score per epoch
          - filepath: string (optional)
            path to save model if savebest is set to True

        outputs:
            None
        '''
        if savebest==True and filepath==None:
            raise Exception("Please enter a path to save the network")

        validation_size = len(validation_data[0])

        print('training network on %i documents, validating on %i documents' \
              % (len(data), validation_size))

        #track best model for saving
        prevbest = 0
        for i in range(epochs):
            correct = 0.
            start = time.time()

            #train
            for doc in range(len(data)):
                inputval = self._list_to_numpy(data[doc])
                feed_dict = {
                    self.doc_input:inputval,
                    self.labels:labels[doc],
                    self.dropout:self.dropout_keep
                }
                pred, cost, _ = self.sess.run(
                    [self.prediction,self.loss,self.optimizer],
                    feed_dict=feed_dict
                )
                assert not np.isnan(cost)
                if np.argmax(pred) == np.argmax(labels[doc]):
                    correct += 1
                sys.stdout.write("epoch %i, sample %i of %i, loss: %f      \r"\
                                 % (i+1,doc+1,len(data),cost))
                sys.stdout.flush()

                #checkpoint periodically 
                if (doc+1) % 50000 == 0:
#               if (doc+1) % 1000 == 0:
                    print("\ntraining time: %.2f" % (time.time()-start))
                    score = self.score(validation_data[0],validation_data[1])
                    print("iteration %i validation accuracy: %.4f%%" % (doc+1, score*100))

                    #reset timer
                    start = time.time()

                    #save if performance better than previous best
                    if savebest and score >= prevbest:
                        prevbest = score
                        self.save(filepath)

            print()
            trainscore = correct/len(data)
            print("epoch %i training accuracy: %.4f%%" % (i+1, trainscore*100))

    def predict(self,data):
        '''
        return the one-hot-encoded predicted labels for given data

        parameters:
          - data: numpy array
            3d numpy array (doc x sentence x word ids) of input data

        outputs:
            numpy array of one-hot-encoded predicted labels for input data
        '''
        labels = []
        for doc in range(len(data)):
            inputval = self._list_to_numpy(data[doc])
            feed_dict = {self.doc_input:inputval,self.dropout:1.0}
            prob = self.sess.run(self.prediction,feed_dict=feed_dict)
            prob = np.squeeze(prob,0)
            one_hot = np.zeros_like(prob)
            one_hot[np.argmax(prob)] = 1
            labels.append(one_hot)

        labels = np.array(labels)
        return labels

    def score(self,data,labels):
        '''
        return the accuracy of predicted labels on given data
        parameters:
          - data: numpy array
            3d numpy array (doc x sentence x word ids) of input data
          - labels: numpy array
            2d numpy array of one-hot-encoded labels

        outputs:
            float representing accuracy of predicted labels on given data
        '''
        correct = 0.
        for doc in range(len(data)):
            inputval = self._list_to_numpy(data[doc])
            feed_dict = {self.doc_input:inputval,self.dropout:1.0}
            prob = self.sess.run(self.prediction,feed_dict=feed_dict)
            if np.argmax(prob) == np.argmax(labels[doc]):
                correct +=1

        accuracy = correct/len(labels)
        return accuracy

    def save(self,filename):
        '''
        save the model weights to a file

        parameters:
          - filepath: string path to save model weights

        outputs:
            None
        '''
        self.saver.save(self.sess,filename)

    def load(self,filename):
        '''
        load model weights from a file

        parameters:
          - filepath: string path from which to load model weights

        outputs:
            None
        '''
        self.saver.restore(self.sess,filename)

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

def tfrecord_loader(data_dir, filenames, nrecords=None, collectors=None):
    feature_list, label_list = collectors
    raw_dataset = tf.data.TFRecordDataset(filenames)
    parsed_dataset = raw_dataset.map(parse_fn)

    for count, ((raw_text, breaks), labels) in enumerate(parsed_dataset):
        if count % 100 == 0:
            sys.stdout.write(f"processing record {count} of {nrecords}             \r")
            sys.stdout.flush()

        # ????????????????????????????????????????????????????????????????????
        # if count == 10000:
        #    break
        # ????????????????????????????????????????????????????????????????????

        sentence_org = 0
        sentences = []
        np_raw_text = np.array(raw_text)                # this is crucial
        np_breaks = np.array(breaks)                    # this is crucial 
        np_labels = np.array(labels)                    # this is crucial 

        assert sum(np_labels) == 1

        for sentence_break in np_breaks:
            sentence = np_raw_text[sentence_org : sentence_org + sentence_break]
            sentences.append(sentence)
            sentence_org = sentence_break

        feature_list.append(sentences)
        label_list.append(np_labels)

##_____________________________________________________________________________
def get_filenames(data_dir, is_training=True, fmt='tfrecords', prefix=''):
    """Return filenames for dataset."""
    if is_training:
        tfrecords = glob.glob(os.path.join(data_dir, prefix + 'train*.%s' % fmt))
    else:
        tfrecords = glob.glob(os.path.join(data_dir, prefix + 'test*.%s' % fmt))

    return tfrecords

##-----------------------------------------------------------------------------        
def main(train=False, eval=False, data_dir=None, model_dir=None, epochs=None, inpfx=''):

    file_prefix = inpfx
    if file_prefix:
        file_prefix += '-'

    # load saved files
    print("loading data")
    vocab = np.load('data/yelp16_embeddings.npy')

    # load pre-processed data
    with open(file_prefix + 'HCAN-metadata.json', 'r') as f:
        metadata = json.load(f)

    params = {}
    params['max_review_words'] = metadata['max_review_words']
    params['max_review_sentences'] = metadata['max_review_sentences']
    params['max_sentence_words'] = metadata['max_sentence_words']
    params['nbr_classes'] = metadata['classes']

    train_set_size = metadata['train_count']
    test_set_size = metadata ['test_count']

    params['model_dir'] = model_dir
    params['data_dir'] = data_dir
    params['vocab'] = vocab
    params['attention_heads'] = 8
    params['attention_size'] = 512
    params['dropout_keep'] = 0.9
    params['activation'] = tf.nn.elu
    params['file_prefix'] = file_prefix
    params['activation'] = tf.nn.elu

    #recombine train and validation datasets into a single memory-resident structure 
    reviews = []
    labels = []
    train_filename = get_filenames(data_dir, is_training=True, prefix=file_prefix)
    valid_filename = get_filenames(data_dir, is_training=False, prefix=file_prefix)
    filenames = [train_filename, valid_filename]
    nbr_records = train_set_size + test_set_size

    with eager_mode():
        tfrecord_loader(data_dir, filenames, nrecords=nbr_records, collectors=(reviews, labels))

    #--------------------------------------------------------------------------        
    #test train split
    print("partitioning training, validation, test data")

    x_train, x_test, y_train, y_test = train_test_split(
        reviews, labels,
        test_size=0.1,
        random_state=1234,
        stratify=labels
    )

    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train, y_train,
        test_size=0.12,
        stratify=y_train
    )

    #create directory for saved model
    if not os.path.exists('./savedmodels'):
        os.makedirs('./savedmodels')

    #train nn
    print("building convolutional attention network")
    classes = params['nbr_classes']
    max_review_sentences = params['max_review_sentences']
    max_sentence_words = params['max_sentence_words']

    nn = hcan(vocab, classes, max_review_sentences, max_sentence_words)

    nn.train(
        x_train, y_train,
        epochs=epochs,
        validation_data=(x_valid, y_valid),
        savebest=True,
        filepath='savedmodels/hcan_yelp16.ckpt'
    )

    #load best nn and test
    nn.load('savedmodels/hcan_yelp16.ckpt')
    score = nn.score(x_test, y_test)
    print("final test accuracy: %.4f%%" % (score * 100))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # candle-like                    
    parser.add_argument('--model_dir',
                        default=os.path.join(MYPATH, 'hcan2_model_dir'),
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
                        default=3,
                        type=int,
                        help='Epochs (converted to estimator steps.')

    parser.add_argument('--inpfx',
                        default='',
                        type=str,
                        help='prefix prepended to tfrecord, JSON and log files')

    args = vars(parser.parse_args())
    main(**args)

