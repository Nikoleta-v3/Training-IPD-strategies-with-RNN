import numpy as np

import axelrod as axl
from axelrod.random_ import random_choice
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential

C, D = axl.Action.C, axl.Action.D



class LSTMPlayer(axl.Player):
    name = "The LSTM homie"
    classifier = {
        "memory_depth": float("inf"),
        "stochastic": True,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(self, model, reshape_history_funct, opening_probability=0.78):
        self.model = model
        self.opening_probability = opening_probability
        self.reshape_history_function = reshape_history_funct
        super().__init__()
        if opening_probability in [0, 1]:
            self.classifier["stochastic"] = False

    def strategy(self, opponent):
        if len(self.history) == 0:
            return random_choice(self.opening_probability)

        history = [action.value for action in opponent.history]
        prediction = float(
            self.model.predict(self.reshape_history_function(history))[0][-1]
        )

        return axl.Action(round(prediction))

    def __repr__(self):
        return self.name


def reshape_history_lstm_model(history):
    return np.array(history).reshape(1, len(history), 1)


def read_model_lstm(filename, num_hidden_cells=100, drop_out_rate=0.2):
    model = Sequential()
    model.add(
        LSTM(num_hidden_cells, return_sequences=True, input_shape=(None, 1))
    )
    model.add(Dropout(rate=drop_out_rate))
    model.add(Dense(1, activation="sigmoid"))
    model.load_weights(filename)

    return model


def read_model_lstm_unknown_length(
    filename, num_hidden_cells=100, drop_out_rate=0.2
):
    model = Sequential()
    model.add(
        LSTM(num_hidden_cells, return_sequences=True, input_shape=(None, 1))
    )
    model.add(LSTM(num_hidden_cells))
    model.add(Dropout(rate=drop_out_rate))
    model.add((Dense(1, activation="sigmoid")))
    model.load_weights(filename)

    return model
