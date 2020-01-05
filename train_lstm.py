import os.path
import sys

import numpy as np

import keras
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras import backend
from keras.callbacks import ModelCheckpoint
from keras.layers import (
    LSTM,
    Dense,
    Dropout,
    TimeDistributed,
    CuDNNLSTM,
    Bidirectional,
)
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split


def batch_generator(inputs, outputs):
    while True:
        for size in range(1, 205):
            batches = [
                (sequence, target)
                for sequence, target in zip(inputs, outputs)
                if len(sequence) == size
            ]

            x, y = zip(*batches)
            batch = np.array(x)
            output_batch = np.array(y)

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

            yield batch, output_batch


def format_sequences_to_input(sequences):
    inputs = sequences.drop(columns=["opponent", "gene_204"]).values
    max_length = len(inputs[0])

    prep_X_train = []
    for histories in range(1, max_length + 1):
        for sequence in inputs:
            assert len(sequence) == max_length
            prep_X_train.append(sequence[:histories])

    return np.array(prep_X_train)


def format_sequences_to_output(sequences):
    inputs = sequences.drop(columns=["opponent", "gene_0"]).values
    max_length = len(inputs[0])

    prep_y_train = []
    for histories in range(1, max_length + 1):
        for sequence in inputs:
            assert len(sequence) == max_length
            prep_y_train.append(sequence[:histories])

    return np.array(prep_y_train)

if __name__ == "__main__":

    experiment = sys.argv[1]

    number_of_epochs = 1000
    num_hidden_cells = 100
    drop_out_rate = 0.2
    num_of_steps = 205

    run_count_filename = "count_run.tex"
    folder_name = f"output_{experiment}"
    file_name = f"{folder_name}/weights-over-time.h5"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        run = 1
        with open(f"{folder_name}/{run_count_filename}", "w") as textfile:
            textfile.write(f"{run}")
    else:
        with open(f"{folder_name}/{run_count_filename}", "r") as textfile:
            run = int(textfile.read()) + 1
        with open(f"{folder_name}/{run_count_filename}", "w") as textfile:
            textfile.write(f"{run}")

    outputs = pd.read_csv("targets.csv", index_col=0)
    y = format_sequences_to_output(outputs)
    sequences = pd.read_csv("sequences.csv", index_col=0)
    inputs = format_sequences_to_input(sequences)
    input_train, input_test, output_train, output_test = train_test_split(
        inputs, y, test_size=0.2, random_state=0
    )

    trainGen = batch_generator(input_train, output_train)
    testGen = batch_generator(input_test, output_test)

    model = Sequential()

    model.add(
            CuDNNLSTM(
                num_hidden_cells, return_sequences=True, input_shape=(None, 1)
            )
        )

    model.add(Dropout(rate=drop_out_rate))

    model.add(Dense(1, activation="sigmoid"))

    if os.path.exists(file_name):
        model.load_weights(file_name)

    model.compile(
        loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    checkpoint = ModelCheckpoint(
        file_name,
        verbose=1,
        monitor="val_loss",
        save_weights_only=True,
        mode="auto",
    )

    history = model.fit_generator(
        trainGen,
        steps_per_epoch=204,
        epochs=number_of_epochs,
        verbose=1,
        validation_data=testGen,
        validation_steps=204,
        callbacks=[checkpoint],
    )

    model.save(f"{folder_name}/final_lstm_model_{run}.h5")

    # Export Evaluation Measuress
    writing_label = f"{num_hidden_cells}_{run}"
    measures = ["acc", "val_acc", "loss", "val_loss"]

    data = list(zip(*[history.history[measure] for measure in measures]))
    df = pd.DataFrame(data, columns=measures)
    df.to_csv(f"{folder_name}/validation_measures_run_{writing_label}.csv")
