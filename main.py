import itertools
import sys

import axelrod as axl
import numpy as np

import sequence_sensei as ss

opponents = [s for s in axl.strategies if s.classifier['long_run_time'] == False]

number_of_generations = 2000
bottlenecks, mutation_probability, half_size_of_populations = [10, 20], [0.05, 0.1], [10, 15, 20]
sequence_length = 205
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
    num_process = int(sys.argv[1])
    opponents_with_seed = get_opponent_seed_combinations(opponents)
    experiments = list(itertools.product(opponents_with_seed, half_size_of_populations,
                                         bottlenecks, mutation_probability))

    for opponent, half_size_of_population, bottleneck, mutation_probability in experiments:

        _ = ss.evolve(opponent=opponent[0],
                      number_of_generations=number_of_generations,
                      bottleneck=bottleneck,
                      mutation_probability=mutation_probability,
                      sequence_length=sequence_length,
                      half_size_of_population=half_size_of_population,
                      seed=opponent[1],
                      num_process=num_process)
