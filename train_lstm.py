import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
import keras
from keras.layers import LSTM, Dense, Dropout, TimeDistributed
from keras.models import Sequential
from sklearn.model_selection import train_test_split


def get_batch(X, Y, sequence_len=205):
    while True:
        for max_length in range(1, sequence_len):
            inputs, outputs = [], []
            n_sequences = len(X)
            for n in range(n_sequences):
                x = np.array(X.iloc[n].values[:max_length])
                x = x.reshape((max_length, 1))

                y = Y.iloc[n].values[:max_length]
                y = y.reshape((max_length, 1))

                inputs.append(x), outputs.append(y)
            yield np.array(inputs), np.array(outputs)


if __name__ == "__main__":

    num_epochs = 100
    num_hidden_cells = 100
    drop_out_rate = 0.2
    num_cores = 4

    config = tf.ConfigProto(device_count={"CPU": num_cores})
    keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

    targets = pd.read_csv("targets.csv", index_col=0)
    targets = targets.drop(columns=["index", "opponent"])

    sequences = pd.read_csv("sequences.csv", index_col=0)

    X_train, X_test, y_train, Y_test = train_test_split(
        sequences, targets, test_size=0.33, random_state=13
    )

    model = Sequential()

    model.add(
        LSTM(num_hidden_cells, return_sequences=True, input_shape=(None, 1))
    )

    model.add(Dropout(rate=drop_out_rate))

    model.add(TimeDistributed(Dense(1, activation="sigmoid")))

    model.compile(
        loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    history = model.fit_generator(
        get_batch(X_train, y_train),
        steps_per_epoch=len(X_train) * 204 // len(X_train),
        epochs=100,
        validation_data=get_batch(X_test, Y_test),
        validation_steps=len(X_test) * 204 // len(X_test),
    )

    model.save("lstm_model.h5")
    model.save_weights("lstm_model_weights.h5")

    # Accuracy plot
    fig, ax = plt.subplots()

    plt.plot(
        history.history["acc"], label="accuracy", color="red", linestyle="--"
    )
    plt.plot(history.history["val_acc"], label=" validation accuracy")

    plt.legend()
    plt.savefig("accuracy_plot.pdf")

    # Loss plot
    fig, ax = plt.subplots()

    plt.plot(history.history["loss"], label="loss", color="red", linestyle="--")
    plt.plot(history.history["val_loss"], label=" validation loss")

    plt.legend()
    plt.savefig("loss_plot.pdf")