import sys

import numpy as np

import axelrod as axl
import sequence_sensei as ss

opponents = [s for s in axl.basic_strategies]

number_of_generations, bottleneck, mutation_probability, half_size_of_population = 20, 10, 0.1, 20
sequence_length = 200
number_of_seeds = 10

def get_opponent_seed_combinations(opponents):
    experiment_values = []
    for opponent in opponents:
        if opponent.classifier['stochastic'] == False:
            experiment_values.append((opponent, np.NaN))
        else:
            for seed in range(number_of_seeds):
                experiment_values.append((opponent, seed))
    return experiment_values

if __name__ == '__main__':
    # add any arguments we want as an input
    experiments = get_opponent_seed_combinations(opponents)
    for opponent, seed in experiments:
        _ = ss.evolve(opponent=opponent, number_of_generations=number_of_generations,
                      bottleneck=bottleneck, mutation_probability=mutation_probability,
                      sequence_length=sequence_length, half_size_of_population=half_size_of_population,
                      seed=seed)