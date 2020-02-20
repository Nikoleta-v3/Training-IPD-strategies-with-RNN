import imp
import os
import random
import sys

import dask
import numpy as np
import pandas as pd

import axelrod as axl

player_class = imp.load_source("player_class", "player_class.py")


if __name__ == "__main__":

    seed_index = int(sys.argv[1])
    filename = sys.argv[2]
    model_type = sys.argv[3]
    data_type = sys.argv[4]

    folder_name = f"meta_tournament_results_{data_type}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    min_size = 5
    max_size = 10
    turns = 200
    repetitions = 50

    if model_type == "sequence":
        model = player_class.read_model_sequence_to_sequence(filename)
    if model_type == "classification":
        model = player_class.read_model_classification(filename)

    for seed in range((seed_index - 1) * 10, seed_index * 10):

        axl.seed(seed)
        size = random.randint(min_size, max_size)
        to_compete = sorted(
            list(set(axl.strategies) - set(axl.long_run_time_strategies)),
            key=lambda x: x.name,
        )
        strategies = random.sample(to_compete, size)

        for opening_probability in [0, 1, 0.78]:

            player = player_class.LSTMPlayer(
                model,
                player_class.reshape_history_lstm_model,
                opening_probability,
            )

            players = [s() for s in strategies] + [player]

            tournament = axl.Tournament(
                players, turns=turns, repetitions=repetitions
            )
            run = dask.delayed(tournament.play)()
            output = dask.compute(run, num_workers=1)

            results = output[0]

            df = pd.DataFrame(results.summarise())
            df["turns"] = turns
            df["repetitions"] = repetitions
            df["size"] = size
            df["seed"] = seed

            df["eigenjesus"] = results.eigenjesus_rating
            df["eigenmoses"] = results.eigenmoses_rating
            df["good_partner_rating"] = results.good_partner_rating
            df["initial_cooperation_rate"] = results.initial_cooperation_rate
            df["median_vengeful_cooperation"] = np.median(
                [cooperation for cooperation in results.vengeful_cooperation]
            )

            df.to_csv(
                f"{folder_name}/result_summary_{model_type}_seed_{seed}_opening_{opening_probability}.csv"
            )

            payoff_matrix = np.array(results.payoff_matrix)
            strategies_names = [str(player) for player in players]
            df_payoff_matrix = pd.DataFrame(
                payoff_matrix, columns=strategies_names, index=strategies_names
            )

            df_payoff_matrix.to_csv(
                f"{folder_name}/payoff_matrix_{model_type}_seed_{seed}_opening_{opening_probability}.csv"
            )
