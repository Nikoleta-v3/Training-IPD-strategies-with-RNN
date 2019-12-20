import sys

import numpy as np

import keras
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split


def get_data(input_file, output_file, test_size=0.33):
    inputs = pd.read_csv(input_file, index_col=0)
    outputs = pd.read_csv(output_file, index_col=0)

    inputs = inputs.drop(columns=["opponent", "gene_204"]).values
    outputs = outputs.drop(columns=["opponent", "gene_0"]).values

    X_train, X_test, y_train, y_test = train_test_split(
        inputs, outputs, test_size=test_size
    )

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))

    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], 1))

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    num_hidden_layers = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    num_of_epochs = int(sys.argv[3])

    num_cores = 1

    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=num_cores,
        inter_op_parallelism_threads=num_cores,
    )

    tf.set_random_seed(0)
    session = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    keras.backend.set_session(session)

    X_train, X_test, y_train, y_test = get_data("sequences.csv", "targets.csv")

    model = Sequential()

    model.add(
        LSTM(
            num_hidden_layers,
            return_sequences=True,
            input_shape=(None, X_train.shape[2]),
        )
    )
    model.add(Dropout(rate=0.3))
    model.add(Dense(1, activation="sigmoid"))

    adam = keras.optimizers.Adam(
        lr=0.0005, beta_1=0.9, beta_2=0.999, amsgrad=False
    )

    model.compile(
        loss="binary_crossentropy", optimizer=adam, metrics=["accuracy"]
    )

    history = model.fit(
        x=X_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_of_epochs,
        verbose=1,
        shuffle=True,
        validation_data=(X_test, y_test),
    )

    writing_label = "%s_%s_%s" % (num_hidden_layers, batch_size, num_of_epochs)
    model.save("output/lstm_model_%s.h5" % writing_label)
    model.save_weights("output/lstm_model_weights_%s.h5" % writing_label)

    # Export evaluation measures
    measures = ["acc", "val_acc", "loss", "val_loss"]

    data = list(zip(*[history.history[measure] for measure in measures]))
    df = pd.DataFrame(data, columns=measures)
    df.to_csv("output/validation_measures_%s.csv" % writing_label)
