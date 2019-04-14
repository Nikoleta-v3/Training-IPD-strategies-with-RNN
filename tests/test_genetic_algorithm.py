import random
import shutil

import numpy as np
import pandas as pd

import axelrod as axl
import sequence_sensei as ss


def test_crossover():
    sequence_one = [random.randint(0, 2) for _ in range(10)]
    sequence_two = [random.randint(0, 2) for _ in range(10)]

    axl.seed(0)
    kid = ss.crossover(sequence_one, sequence_two)
    assert kid == sequence_one[:6]+ sequence_two[6:]

def test_mutation():
    mutation_probability = 0.3
    sequence = [1, 0, 1, 0, 1, 1, 0, 0]

    axl.seed(1)
    mutated_sequence = [ss.mutation(gene, mutation_probability) for gene in sequence]
    assert mutated_sequence == [0] + sequence[1:3] + [1] + sequence[4:]

def test_evolve():
    player = axl.Cooperator
    number_of_generations = 5
    bottleneck = 10
    mutation_probability = 0.1
    sequence_length = 10
    half_size_of_population = 10
    seed = np.NaN

    score, best_sequence = ss.evolve(opponent=player,
                                     number_of_generations=number_of_generations,
                                     bottleneck=bottleneck,
                                     mutation_probability=mutation_probability,
                                     sequence_length=sequence_length,
                                     half_size_of_population=half_size_of_population,
                                     seed=seed)

    result = pd.read_csv('raw_data/Cooperator_nan/main.csv')
    assert list(result.columns) == ['opponent', 'seed', 'num. of generations', 'bottleneck',
                                    'mutation probability', 'half size population',
                                    'generation', 'index', 'score', 'gene_0', 'gene_1',
                                    'gene_2', 'gene_3', 'gene_4', 'gene_5', 'gene_6',
                                    'gene_7', 'gene_8', 'gene_9']
    assert list(result['generation'].unique()) == [0, 1, 2, 3, 4, 5]
    assert score == 5.0
    assert best_sequence == [0 for _ in range(sequence_length)]

    assert list(result['opponent'].unique())[0] == 'Cooperator'
    assert int(result['num. of generations'].unique()[0]) == 5
    assert int(result['bottleneck'].unique()[0]) == 10
    assert float(result['mutation probability'].unique()[0]) == 0.1
    assert int(result['half size population'].unique()[0]) == 10

    shutil.rmtree('raw_data/Cooperator_nan')
