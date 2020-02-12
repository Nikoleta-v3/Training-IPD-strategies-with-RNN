import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import keras
from keras.callbacks import ModelCheckpoint
from keras.layers import (
    LSTM,
    Bidirectional,
    CuDNNLSTM,
    Dense,
    Dropout,
    RepeatVector,
    TimeDistributed,
)
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing import sequence

if __name__ == "__main__":

    experiment = sys.argv[1]

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

    inputs = pd.read_csv("padded_inputs_classification.csv", index_col=0)
    outputs = pd.read_csv("classification_output.csv", index_col=0)

    X = inputs.values
    y = outputs["target"].values

    data = list(zip(X, y))
    train, test = train_test_split(data, test_size=0.2)
    X_train, y_train = list(zip(*train))
    X_test, y_test = list(zip(*test))

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    max_length = len(X[0])
    batch_size = 128
    number_of_epochs = 100

    num_cells = 100
    drop_out_rate = 0.2

    top_words = 3
    embedding_vector_length = 1

    model = Sequential()

    model.add(
        Embedding(top_words, embedding_vector_length, input_length=max_length)
    )
    model.add(CuDNNLSTM(num_cells))

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

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=number_of_epochs,
        verbose=3,
        batch_size=batch_size,
        callbacks=[checkpoint],
    )

    model.save(f"{folder_name}/final_lstm_model_{run}.h5")

    measures = ["acc", "val_acc", "loss", "val_loss"]

    data = list(zip(*[history.history[measure] for measure in measures]))
    df = pd.DataFrame(data, columns=measures)
    df.to_csv(f"{folder_name}/validation_measures_{batch_size}_{run}.csv")
