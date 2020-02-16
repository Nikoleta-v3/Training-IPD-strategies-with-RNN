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
    filename = sys.argv[
        2
    ]
    model_type = sys.argv[3]
    num_processes = int(sys.argv[4])

    folder_name = "meta_tournament_results"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    min_size = 5
    max_size = 10
    turns = 205
    repetitions = 50

    if model_type == "sequence":
        model = player_class.read_model_sequence_to_sequence(filename)
    if model_type == "classification":
        model = player_class.read_model_classification(filename)

    for seed in range(max_seed):

        player = player_class.LSTMPlayer(
            model, player_class.reshape_history_lstm_model
        )

        axl.seed(seed)
        size = random.randint(min_size, max_size)
        to_compete = list(
            set(axl.strategies) - set(axl.long_run_time_strategies)
        )
        strategies = random.sample(to_compete, size)

        players = [s() for s in strategies] + [player]
        turns = turns
        repetitions = repetitions

        tournaments = axl.Tournament(
            players, turns=turns, repetitions=repetitions
        )
        results = tournaments.play(processes=num_processes)

        df = pd.DataFrame(results.summarise())
        df["turns"] = turns
        df["repetitions"] = repetitions

        df["eigenjesus"] = results.eigenjesus_rating
        df["eigenmoses"] = results.eigenmoses_rating
        df["good_partner_rating"] = results.good_partner_rating
        df["initial_cooperation_rate"] = results.initial_cooperation_rate
        df["median_vengeful_cooperation"] = np.median(
            [cooperation for cooperation in results.vengeful_cooperation]
        )

        df.to_csv(f"meta_tournament_results/result_summary_{model_type}_seed_{seed}.csv")
