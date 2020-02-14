import imp
import os
import random
import sys

import numpy as np
import pandas as pd

import axelrod as axl

player_class = imp.load_source("player_class", "player_class.py")


if __name__ == "__main__":

    max_seed = int(sys.argv[1])

    folder_name = "meta_tournament_results"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    min_size = 5
    max_size = 10
    turns = 205
    repetitions = 5

    model = player_class.read_in_model_lstm_unknown_length(
        "hawk_output/output_lstm_unknown_model/weights-over-time.h5"
    )

    for seed in range(max_seed):

        player = player_class.LSTMPlayer(
            model, player_class.reshape_history_lstm_model
        )

        axl.seed(seed)
        size = random.randint(min_size, max_size)
        strategies = random.sample(axl.strategies, size)

        players = [s() for s in strategies] + [player]
        turns = turns
        repetitions = repetitions

        tournaments = axl.Tournament(
            players, turns=turns, repetitions=repetitions
        )
        results = tournaments.play()

        df = pd.DataFrame(results.summarise())
        df["turns"] = turns
        df["repetitions"] = repetitions

        df["eigenjesus"] = results.eigenjesus_rating
        df["eigenmoses"] = results.eigenmoses_rating
        df["initial_cooperation_rate"] = results.initial_cooperation_rate
        df["median_vengeful_cooperation"] = np.median(
            [cooperation for cooperation in results.vengeful_cooperation]
        )

        df.to_csv(f"meta_tournament_results/result_summary_seed_{seed}.csv")