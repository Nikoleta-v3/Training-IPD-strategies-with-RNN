import random
import axelrod as axl
import numpy as np
import sequence_sensei as ss

C, D = axl.Action

def test_get_sequence_str():

    assert ss.get_sequence_str([C, C, C]) == 'CCC'
    assert ss.get_sequence_str([D, D, D]) == 'DDD'

    axl.seed(0)
    sequence = [axl.Action(random.randint(0, 2)) for _ in range(4)]
    assert ss.get_sequence_str(sequence) == 'CCDC'

def test_get_fitness_of_individual():

    sequence = [C for _ in range(10)]
    seed = np.NaN
    i, score_vs_c = ss.get_fitness_of_individual(sequence, axl.Cooperator, seed=seed,
                                            index=0, turns=10).compute()
    assert i == 0
    assert score_vs_c == 3

    i, score_vs_d = ss.get_fitness_of_individual(sequence, axl.Defector, seed=seed,
                                            index=0, turns=10).compute()
    assert i == 0
    assert score_vs_d == 0

def test_get_fitness_of_population():
    