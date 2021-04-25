from sklearn.model_selection import train_test_split
import os
from model import create_model
from utils import encoder_data, add_noise
from hparams import BATCH_SIZE
import numpy as np
from tensorflow.keras.callbacks import Callback, ModelCheckpoint


def generate_data(data, batch_size):
    cur_index = 0
    while True:
        x, y = [], []
        for i in range(batch_size):
            y.append(encoder_data(data[cur_index]))
            x.append(encoder_data(add_noise(data[cur_index], 0.94, 0.985)))
            cur_index += 1
            if cur_index > len(data) - 1:
                cur_index = 0
        yield np.array(x), np.array(y)


def main():
    train_data, valid_data = train_test_split(list_ngrams, test_size=0.2, random_state=42)
    train_generator = generate_data(train_data, batch_size=BATCH_SIZE)
    validation_generator = generate_data(valid_data, batch_size=BATCH_SIZE)

    model = create_model()
    checkpointer = ModelCheckpoint(filepath=os.path.join('./model/spell_{val_acc:.2f}.h5'), save_best_only=True,
                                   verbose=1)
    model.fit_generator(train_generator, steps_per_epoch=len(train_data) // BATCH_SIZE, epochs=10,
                        validation_data=validation_generator, validation_steps=len(valid_data) // BATCH_SIZE,
                        callbacks=[checkpointer])
