import imp

import numpy as np

from keras.models import load_model

player_class = imp.load_source("player_class", "player_class.py")


def test_init_for_deterministic_player_that_cooperates():
    opening_probability = 1

    model = load_model("basic/output_lstm/model-over-time.h5")
    player = player_class.LSTMPlayer(
        model,
        opening_probability=opening_probability,
        reshape_history_funct=player_class.reshape_history_simple_model,
    )

    assert player.opening_probability == opening_probability
    assert player.classifier["stochastic"] == False


def test_init_for_deterministic_player_that_defects():
    opening_probability = 0

    model = load_model("basic/output_lstm/model-over-time.h5")
    player = player_class.LSTMPlayer(
        model,
        opening_probability=opening_probability,
        reshape_history_funct=player_class.reshape_history_simple_model,
    )

    assert player.opening_probability == opening_probability
    assert player.classifier["stochastic"] == False


def test_init_for_stochastic_player():
    opening_probability = 0.5

    model = load_model("basic/output_lstm/model-over-time.h5")
    player = player_class.LSTMPlayer(
        model,
        opening_probability=opening_probability,
        reshape_history_funct=player_class.reshape_history_simple_model,
    )

    assert player.opening_probability == opening_probability
    assert player.classifier["stochastic"] == True


def test_reshape_history():
    length = 10
    history = np.array([0 for _ in range(length)])

    assert player_class.reshape_history_simple_model(history).shape == (
        1,
        length,
        1,
    )
