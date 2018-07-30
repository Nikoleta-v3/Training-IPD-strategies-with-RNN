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
    population = ss.get_initial_population(5)
    scores = ss.get_fitness_of_population(population, axl.TitForTat, turns=205, seed=0)
    assert scores == [(0, 1.0439024390243903),
                      (1, 2.9902439024390244),
                      (2, 1.5414634146341464),
                      (3, 2.4878048780487805),
                      (4, 2.029268292682927),
                      (5, 2.0),
                      (6, 2.5170731707317073),
                      (7, 1.5121951219512195),
                      (8, 3.0),
                      (9, 1.0195121951219512)]

def test_get_fitness_of_population_with_random_opponent():
    population = ss.get_initial_population(5)
    scores = ss.get_fitness_of_population(population, axl.Random, turns=205, seed=0)

    assert scores == [(0, 2.8780487804878048),
                      (1, 1.4341463414634146),
                      (2, 2.502439024390244),
                      (3, 1.8097560975609757),
                      (4, 2.180487804878049),
                      (5, 2.131707317073171),
                      (6, 1.8048780487804879),
                      (7, 2.5073170731707317),
                      (8, 1.4195121951219511),
                      (9, 2.892682926829268)]