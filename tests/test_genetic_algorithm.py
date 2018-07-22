import sequence_sensei as ss
import axelrod as axl
import random

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
    pass