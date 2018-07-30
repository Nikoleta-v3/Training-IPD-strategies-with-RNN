import sequence_sensei as ss
import random

def test_initial_population():
    half_size_of_population = 20
    sequence_length = 20
    initial_population = ss.get_initial_population(half_size_of_population, sequence_length=sequence_length)

    assert len(initial_population) == 40

    individual = random.randint(0, 20)
    assert len(initial_population[individual]) == sequence_length
    assert initial_population[-1] == [0 for _ in range(20)]
    assert initial_population[-2] == [1 for _ in range(20)]
