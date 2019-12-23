import sys
import os

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
    experiment = sys.argv[4]

    num_cells = 200
    num_cores = 1
    drop_out_rate = 0.2
    folder_name = f"output_{experiment}"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=num_cores,
        inter_op_parallelism_threads=num_cores,
    )

    session = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    keras.backend.set_session(session)

    X_train, X_test, y_train, y_test = get_data("sequences.csv", "targets.csv")

    model = Sequential()

    for _ in range(num_hidden_layers):
        model.add(
            LSTM(
                num_cells,
                return_sequences=True,
                input_shape=(None, X_train.shape[2]),
            )
        )

        model.add(Dropout(rate=drop_out_rate))
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
    model.save(f"{folder_name}/lstm_model_{writing_label}.h5")
    model.save_weights(f"{folder_name}/lstm_model_weights_{writing_label}.h5")

    # Export evaluation measures
    measures = ["acc", "val_acc", "loss", "val_loss"]

    data = list(zip(*[history.history[measure] for measure in measures]))
    df = pd.DataFrame(data, columns=measures)
    df.to_csv(f"{folder_name}/validation_measures_{writing_label}.csv")
