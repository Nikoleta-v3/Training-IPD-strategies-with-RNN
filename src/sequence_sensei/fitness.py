"""
A file which contains code to return the score of binary sequences against an opponent
after playing an IPD match.
"""
import axelrod as axl
import numpy as np
import dask

from multiprocessing.pool import ThreadPool

def get_sequence_str(sequence):
    string_sequence = ""
    for action in [axl.Action(i) for i in sequence]:
        string_sequence += str(action)

    return string_sequence

@dask.delayed
def get_fitness_of_individual(sequence, opponent, seed, index=0, turns=205):
    if seed is not np.NaN:
        axl.seed(seed)

    opponent = opponent()
    player = axl.Cycler(get_sequence_str(sequence))
    match = axl.Match([opponent, player], turns=turns)
    _ = match.play()

    return index, match.final_score_per_turn()[-1]

def get_fitness_of_population(population, opponent, seed, index, num_process=1):
    index_scores = []
    for index, individual in enumerate(population):
        index_scores.append(get_fitness_of_individual(individual, opponent, seed=seed,
                                                      index=index))

    with dask.config.set(pool=ThreadPool(num_process)):
        result = dask.compute(*index_scores)
    return list(result)