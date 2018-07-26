import sys

import numpy as np

import axelrod as axl
import sequence_sensei as ss

opponents = [s for s in axl.basic_strategies]

number_of_generations, bottleneck, mutation_probability = 20, 10, 0.1
sequence_length, size_of_population = 200, 20
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
    # add any agruments we want as an iput
    experiments = get_opponent_seed_combinations(opponents)
    for opponent, seed in experiments:
        _ = ss.evolve(opponent=opponent, number_of_generations=number_of_generations,
                    bottleneck=bottleneck, mutation_probability=mutation_probability,
                    sequence_length=sequence_length, size_of_population=size_of_population,
                    seed=seed)