"""
A file which contains code to perform a genetic algorithm on binary sequences.
"""
import csv
import os
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

def subset_population(population, indices):
    subset = []
    for i in indices:
        subset.append(population[i])
    return subset

def evolve(opponent, number_of_generations, bottleneck, mutation_probability,
           sequence_length, size_of_population, seed, num_process=1):

    headers = ['generation', 'index', 'score']
    headers += ['gene_{}'.format(i) for i in range(sequence_length)]

    print('|Initialising population|')
    generation = 0
    population = ss.get_initial_population(size_of_population=size_of_population,
                                           sequence_length=sequence_length)
    scores = ss.get_fitness_of_population(population=population, opponent=opponent,
                                         seed=seed, num_process=num_process)

    results = [[generation, *scores[i], *population[i]] for i in range(size_of_population * 2)]
    results.sort(key=lambda tup:tup[1], reverse=True)

    path = 'raw_data/{}_{}'.format(opponent.name, seed)
    os.mkdir(path)
    with open("{}/main.csv".format(path), "a") as data_file:
        data_writer = csv.writer(data_file)
        data_writer.writerow(headers)
        for row in results:
            data_writer.writerow(row)
        print('|Finish Generation 0|')

        while generation < number_of_generations:
            generation += 1
            indices_to_keep = [results[i][1] for i in range(bottleneck)]
            population = subset_population(population, indices_to_keep)

            while len(population) < 2 * size_of_population:
                i, j  = [random.randint(0, bottleneck) for _ in range(2)]
                new_individual = crossover(population[i], population[j])
                new_individual = [mutation(gene, mutation_probability) for gene in new_individual]
                population.append(new_individual)

            scores = ss.get_fitness_of_population(population=population,
                                                  opponent=opponent, seed=seed,
                                                  num_process=num_process)

            results = [[generation, *scores[i], *population[i]] for i in range(size_of_population * 2)]
            results.sort(key=lambda tup:tup[1], reverse=True)
            for row in results:
                data_writer.writerow(row)
            print('|Finish Generation {}|'.format(generation))
            
# TODO: List from indices in list