"""
A file which contains code to perform a genetic algorithm on binary sequences.
"""
import csv
import random
import sequence_sensei as ss

def crossover(sequence_one, sequence_two):
    sequence_length = len(sequence_one)
    crossover_point = random.randint(0, sequence_length)

    return sequence_one[:crossover_point] + sequence_two[crossover_point:]

def mutation(gene, mutation_probability):
    if random.random() < mutation_probability:
        return abs(gene - 1)
    return gene

def evolve(opponent, number_of_generations, bottleneck, mutation_probability,
           sequence_length, size_of_population, seed, index, num_process=1):

    headers = ['generation', 'opponent', 'seed', 'indes']
    headers += ['gene_{}'.format(i) for i in range(sequence_length)] + ['score']

    generation = 0
    population = ss.get_initial_population(size_of_population=size_of_population,
                                           sequence_length=sequence_length)
    score = ss.get_fitness_of_population(population=population, opponent=opponent,
                                         seed=seed, num_process=num_process)
    # combine scores with initial population on index
    results = zip()

    # write file
    # make folder {}_{}.format(opponent.name, seed)
    with open("main.csv", "a") as data_file:
        data_writer = csv.writer(data_file)
        data_writer.writerow(headers)
        data_writer.writerow(results)

    while generation < number_of_generations:
        indices_to_keep = [i for (i, s) in score[0: bottleneck]]
        population = population[indices_to_keep] # this is going to break

        while len(population) < 2 * size_of_population:
            i, j  = [random.randint(0, bottleneck) for _ in range(2)]
            new_individual = crossover(population[i], population[j])
            new_individual = [mutation(gene, mutation_probability) for gene in new_individual]
            population.append(new_individual)
        
        score = ss.get_fitness_of_population(population=population, opponent=opponent,
                                             seed=seed, num_process=num_process)
        # combine scores with initial population on index
        results = zip()
        data_writer.writerow(results)
        generation += 1