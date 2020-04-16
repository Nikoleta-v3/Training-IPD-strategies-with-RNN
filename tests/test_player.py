import imp

import axelrod as axl
import numpy as np

from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential, load_model

C, D = axl.Action.C, axl.Action.D


player_class = imp.load_source("player_class", "player_class.py")


def test_init_for_deterministic_player_that_cooperates():
    opening_probability = 1

    model = load_model("basic/output_lstm/model-over-time.h5")
    player = player_class.LSTMPlayer(
        model,
        opening_probability=opening_probability,
        reshape_history_funct=player_class.reshape_history_lstm_model,
    )

    assert player.opening_probability == opening_probability
    assert player.classifier["stochastic"] == False


def test_init_for_deterministic_player_that_defects():
    opening_probability = 0

    model = load_model("basic/output_lstm/model-over-time.h5")
    player = player_class.LSTMPlayer(
        model,
        opening_probability=opening_probability,
        reshape_history_funct=player_class.reshape_history_lstm_model,
    )

    assert player.opening_probability == opening_probability
    assert player.classifier["stochastic"] == False


def test_init_for_stochastic_player():
    opening_probability = 0.5

    model = load_model("basic/output_lstm/model-over-time.h5")
    player = player_class.LSTMPlayer(
        model,
        opening_probability=opening_probability,
        reshape_history_funct=player_class.reshape_history_lstm_model,
    )

    assert player.opening_probability == opening_probability
    assert player.classifier["stochastic"] == True


def test_reshape_history():
    length = 10
    history = np.array([0 for _ in range(length)])

    assert player_class.reshape_history_lstm_model(history).shape == (
        1,
        length,
        1,
    )


def test_read_model_sequence_to_sequence():
    filename = "hawk_output/output_lstm_model_basic/weights-over-time.h5"

    model = player_class.read_model_sequence_to_sequence(filename)

    assert type(model) == Sequential
    assert len(model.layers) == 3
    assert type(model.layers[0]) == LSTM
    assert type(model.layers[1]) == Dropout


def test_read_model_classification():
    filename = (
        "hawk_output/output_lstm_unknown_model_basic/weights-over-time.h5"
    )

    model = player_class.read_model_classification(filename)

    assert type(model) == Sequential
    assert len(model.layers) == 4
    assert type(model.layers[0]) == LSTM
    assert type(model.layers[1]) == LSTM


def test_init_for_stochastic_player_with_defection_opening():
    opening_probability = 0

    model = load_model("basic/output_lstm/model-over-time.h5")
    player = player_class.StochasticLSTMPlayer(
        model,
        opening_probability=opening_probability,
        reshape_history_funct=player_class.reshape_history_lstm_model,
    )

    assert player.opening_probability == opening_probability
    assert player.classifier["stochastic"] == True


def test_init_for_stochastic_player_with_cooperation_opening():
    opening_probability = 1

    model = load_model("basic/output_lstm/model-over-time.h5")
    player = player_class.StochasticLSTMPlayer(
        model,
        opening_probability=opening_probability,
        reshape_history_funct=player_class.reshape_history_lstm_model,
    )

    assert player.opening_probability == opening_probability
    assert player.classifier["stochastic"] == True


def test_stochastic_lstm_in_matches():
    model = player_class.read_model_sequence_to_sequence(
        "hawk_output/output_lstm_model_basic/weights-over-time.h5"
    )

    player = player_class.StochasticLSTMPlayer(
        model, player_class.reshape_history_lstm_model, opening_probability=1
    )

    match = axl.Match(players=[player, axl.Cooperator()], turns=5)

    axl.seed(0)
    _ = match.play()
    assert match.result == [(C, C), (D, C), (C, C), (C, C), (C, C)]

    axl.seed(1)
    _ = match.play()
    assert match.result == [(C, C), (C, C), (C, C), (C, C), (C, C)]
