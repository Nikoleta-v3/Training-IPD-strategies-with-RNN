"""
A file that contains code to create an initial population of binary sequences.
"""
import numpy as np


def get_initial_population(half_size_of_population, sequence_length=200):
    """
    Generates an initial population of sequences. Note that the length
    of the population which is being generated is 2 * half_size_of_population.
    """
    cuts = np.linspace(1, sequence_length, half_size_of_population, dtype=int)
    sequences = []
    for cut in cuts:
        sequences.append(
            [1 for _ in range(cut)] + [0 for _ in range(sequence_length - cut)]
        )
        sequences.append(
            [0 for _ in range(cut)] + [1 for _ in range(sequence_length - cut)]
        )

    return sequences
