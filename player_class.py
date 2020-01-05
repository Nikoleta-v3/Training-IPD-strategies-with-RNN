import numpy as np

import axelrod as axl
from axelrod.random_ import random_choice

C, D = axl.Action.C, axl.Action.D


class LSTMPlayer(axl.Player):
    name = "The LSTM homie"
    classifier = {
        "memory_depth": float("inf"),
        "stochastic": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(self, model, reshape_history_funct, opening_probability=0.78):
        self.model = model
        self.opening_probability = opening_probability
        self.reshape_history_function = reshape_history_funct
        super().__init__()

    def strategy(self, opponent):
        if len(self.history) == 0:
            return random_choice(self.opening_probability)

        current_length = len(opponent.history)
        history = [action.value for action in opponent.history]

        prediction = self.model.predict(self.reshape_history_function(history))
        prediction = prediction.reshape(1, current_length)

        return axl.Action(round(prediction[-1][0]))

    def __repr__(self):
        return self.name


def reshape_history_simple_model(history):
    return np.array(history).reshape(1, len(history), 1)
