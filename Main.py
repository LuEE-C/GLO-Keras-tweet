# Naive attempt at implementing https://arxiv.org/abs/1707.05776 on whole tweet generation
# I believe that with some soft clipping on the wassertein distance something decent might come out of it

from data_util import tweet_to_npy, predictions_to_string
import numpy as np
from keras.models import Model
from DenseNet import DenseNet
from keras.layers import BatchNormalization, LSTM, Input, Reshape, TimeDistributed, Dense
from keras.callbacks import ModelCheckpoint
import keras.backend as K

import random
import os
import pickle
from time import time

def lstm_model(shape1, shape2, shape3):
    input = Input(shape=(shape1, shape2))
    x = DenseNet(input_tensor=input, nb_layers=6, nb_dense_block=5, growth_rate=12,
             nb_filter=16, dropout_rate=0.5, weight_decay=1E-4, compression_rate=0.5)
    x = Reshape((140, 1))(x)
    x = TimeDistributed(Dense(shape3, activation='softmax'))(x)
    model = Model(inputs=[input], outputs=[x])
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()
    return model


def train(from_save=False, epoch=100, batch_size=512):
    X = np.load('tensors/TweetLatent_0.npy')
    y = np.load('tensors/Tweet_0.npy')

    model = lstm_model(X.shape[1], X.shape[2], y.shape[2])
    del X
    del y

    if from_save:
        model.load_weights('Model')

    losses_over_time = []
    for e in range(epoch):
        start = time()
        iter_items = list(range(len(os.listdir('tensors/'))//2))
        random.shuffle(iter_items)
        for file_index in iter_items:
            X = np.load('tensors/TweetLatent_' + str(file_index) + '.npy')
            y = np.load('tensors/Tweet_' + str(file_index) + '.npy')
            iter_batch = list(range(X.shape[0]//batch_size))
            random.shuffle(iter_batch)
            for batch_num in iter_batch:
                batch_x, batch_y = X[batch_num * batch_size: batch_num * batch_size + 1], y[batch_num * batch_size: batch_num * batch_size + 1]

                losses_over_time.append(model.train_on_batch(batch_x, batch_y))

            print('Epoch', e, 'File', file_index, 'Loss : ', np.mean(np.array(losses_over_time[-X.shape[0]//batch_size:])))
            pickle.dump(losses_over_time, open("losses_over_time.pkl", "wb"))
            model.save_weights('Model')

        print('Epoch', e, 'took :', time() - start)
        predict(e, model, y.shape[1])

def predict(epoch, model, shape):
    X = np.random.normal(size=(5, shape, 10))

    predictions = model.predict(X, batch_size=5)
    predictions = predictions_to_string(predictions)
    with open('prediction_' + str(epoch) + '.txt', 'w') as f:
        f.write(predictions)

if __name__ == '__main__':
    # tweet_to_npy()
    train(from_save=False)
