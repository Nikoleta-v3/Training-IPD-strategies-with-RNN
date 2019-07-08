import numpy as np

import keras
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras import backend
from keras.layers import LSTM, Dense, Dropout, TimeDistributed
from keras.models import Sequential
from sklearn.model_selection import train_test_split


def batch_generator(input_path, output_path, bs=2470):
    while True:
        skip = []
        for iterations in range(0, 204):
            print(iterations)
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

            yield (batch, output_batch)


if __name__ == "__main__":

    num_epochs = 500
    num_hidden_cells = 100
    drop_out_rate = 0.2
    num_cores = 10

    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=num_cores,
        inter_op_parallelism_threads=num_cores,
    )

    tf.set_random_seed(0)
    session = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    keras.backend.set_session(session)

    model = Sequential()

    model.add(
        LSTM(num_hidden_cells, return_sequences=True, input_shape=(None, 1))
    )

    model.add(Dropout(rate=drop_out_rate))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    trainGen = batch_generator("inputs_train.csv", "outputs_train.csv")
    testGen = batch_generator("inputs_test.csv", "outputs_test.csv", bs=1218)

    history = model.fit_generator(
        trainGen,
        steps_per_epoch=204,
        epochs=num_epochs,
        verbose=2,
        validation_data=testGen,
        validation_steps=204,
        use_multiprocessing=True,
        workers=num_cores,
    )

    model.save("output/lstm_model.h5")
    model.save_weights("output/lstm_model_weights.h5")

    # Accuracy plot
    fig, ax = plt.subplots()

    plt.plot(
        history.history["acc"], label="accuracy", color="red", linestyle="--"
    )
    plt.plot(history.history["val_acc"], label=" validation accuracy")

    plt.legend()
    plt.savefig("output/accuracy_plot.pdf")

    # Loss plot
    fig, ax = plt.subplots()

    plt.plot(history.history["loss"], label="loss", color="red", linestyle="--")
    plt.plot(history.history["val_loss"], label=" validation loss")

    plt.legend()
    plt.savefig("output/loss_plot.pdf")
