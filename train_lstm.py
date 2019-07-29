import os.path
import sys

import numpy as np

import keras
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras import backend
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Dense, Dropout, TimeDistributed, CuDNNLSTM
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split


def batch_generator(input_path, output_path, bs=2470, num_of_steps=202):
    while True:
        skip = []
        for iterations in range(0, 202):
            if iterations > 0:
                skip += [
                    x for x in range((iterations - 1) * bs, bs * iterations)
                ]
            batch = pd.read_csv(
                input_path, nrows=bs, skiprows=skip, index_col=0
            ).values

            output_batch = pd.read_csv(
                output_path, nrows=bs, skiprows=skip, index_col=0
            ).values

            batch = np.array(
                [
                    [x for x in mini_batch if np.isnan(x) == False]
                    for mini_batch in batch
                ]
            )
            output_batch = np.array(
                [
                    [x for x in mini_batch if np.isnan(x) == False]
                    for mini_batch in output_batch
                ]
            )

            try:
                batch = batch.reshape((batch.shape[0], batch.shape[1], 1))
                output_batch = output_batch.reshape(
                    (output_batch.shape[0], output_batch.shape[1], 1)
                )
            except IndexError:
                batch = batch.reshape((batch.shape[0], 1, 1))
                output_batch = output_batch.reshape(
                    (output_batch.shape[0], 1, 1)
                )

            if iterations == 201:
                skip = []

            yield (batch, output_batch)


if __name__ == "__main__":

    num_epochs = 150
    num_hidden_cells = 200
    drop_out_rate = 0.2
    num_cores = 10
    num_of_steps = 202

    run_count_filename = 'count_run.txt'
    file_name = "output/weights.best.hdf5"
    with open(run_count_filename, 'r') as textfile:
        run = int(textfile.read().split('= ')[-1])

    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=num_cores,
        inter_op_parallelism_threads=num_cores,
    )

    tf.set_random_seed(0)
    session = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    keras.backend.set_session(session)

    if os.path.isfile(file_name):
        model = load_model(file_name)
    else:
        model = Sequential()

        model.add(
            CuDNNLSTM(num_hidden_cells, return_sequences=True, input_shape=(None, 1))
        )

        model.add(Dropout(rate=drop_out_rate))

        model.add(Dense(1, activation="sigmoid"))
        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

    checkpoint = ModelCheckpoint(
        file_name, monitor='loss', verbose=1, save_best_only=True, mode="max"
    )
    callbacks_list = [checkpoint]

    trainGen = batch_generator("inputs_train.csv", "outputs_train.csv")
    testGen = batch_generator("inputs_test.csv", "outputs_test.csv", bs=1218)

    history = model.fit_generator(
        trainGen,
        steps_per_epoch=num_of_steps,
        epochs=num_epochs,
        verbose=1,
        validation_data=testGen,
        validation_steps=num_of_steps,
        callbacks=callbacks_list,
    )

    # Accuracy plot
    fig, ax = plt.subplots()

    plt.plot(
        history.history["acc"], label="accuracy", color="red", linestyle="--"
    )
    plt.plot(history.history["val_acc"], label=" validation accuracy")

    plt.legend()
    plt.savefig("output/accuracy_plot_%s.pdf" % run)

    # Loss plot
    fig, ax = plt.subplots()

    plt.plot(history.history["loss"], label="loss", color="red", linestyle="--")
    plt.plot(history.history["val_loss"], label=" validation loss")

    plt.legend()
    plt.savefig("output/loss_plot_run_%s.pdf" % run)

    with open(run_count_filename, 'w') as textfile:
        textfile.write('run_number = %s' % (run + 1))