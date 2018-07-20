import numpy as np

def get_initial_population(size_of_population, sequence_length=200):
    cuts = np.linspace(1, sequence_length, size_of_population, dtype = int)
    sequences = []
    for cut in cuts:
        sequences.append([1 for _ in range(cut)] + [0 for _ in range(sequence_length - cut)])
        sequences.append([0 for _ in range(cut)] + [1 for _ in range(sequence_length - cut)])

    return sequences