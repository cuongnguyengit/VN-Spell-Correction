import tensorflow as tf


# Build the neural network
# this is adapted from the seq2seq architecture, which can be used for Machine Translation, Text Summarization Image Captioning ...
from tf.keras.models import Sequential
from tf.keras.layers import Activation, TimeDistributed, Dense, LSTM, Bidirectional
from tf.keras.optimizers import Adam
from symbol import *
from hparams import MA
from tf.keras.utils import plot_model


def create_model():
    encoder = LSTM(256, input_shape=(MAXLEN, len(alphabet)), return_sequences=True)
    decoder = Bidirectional(LSTM(256, return_sequences=True, dropout=0.2))

    model = Sequential()
    model.add(encoder)
    model.add(decoder)
    model.add(TimeDistributed(Dense(256)))
    model.add(Activation('relu'))
    model.add(TimeDistributed(Dense(len(alphabet))))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy'])

    model.summary()
    plot_model(model, to_file='model.png')
    return model
