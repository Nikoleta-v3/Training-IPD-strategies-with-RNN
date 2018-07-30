"""
A file which contains code to perform a genetic algorithm on binary sequences.
"""
import csv
import os
import random

import tqdm

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
           sequence_length, half_size_of_population, seed, num_process=1):

    headers = ['generation', 'index', 'score']
    headers += ['gene_{}'.format(i) for i in range(sequence_length)]

    generation = 0
    population = ss.get_initial_population(half_size_of_population=half_size_of_population,
                                           sequence_length=sequence_length)
    scores = ss.get_fitness_of_population(population=population, opponent=opponent,
                                          seed=seed, turns=sequence_length, num_process=num_process)

    results = [[generation, *scores[i], *population[i]] for i in range(half_size_of_population * 2)]
    results.sort(key=lambda tup:tup[2], reverse=True)

    path = 'raw_data/{}_{}'.format(opponent.name, seed)
    if not os.path.exists(path):
        os.mkdir(path)
    with open("{}/main.csv".format(path), "w") as data_file:
        data_writer = csv.writer(data_file)
        data_writer.writerow(headers)
        for row in results:
            data_writer.writerow(row)

        pbar = tqdm.tqdm(total=number_of_generations)
        while generation < number_of_generations:
            generation += 1
            indices_to_keep = [results[i][1] for i in range(bottleneck)]
            new_population = subset_population(population, indices_to_keep)
            population = new_population

            while len(population) < 2 * half_size_of_population:
                i, j  = [random.randint(0, bottleneck - 1) for _ in range(2)]
                new_individual = crossover(population[i], population[j])
                new_individual = [mutation(gene, mutation_probability) for gene in new_individual]
                population.append(new_individual)

            scores = ss.get_fitness_of_population(population=population,
                                                  opponent=opponent, seed=seed,
                                                  turns=sequence_length,
                                                  num_process=num_process)

            results = [[generation, *scores[i], *population[i]] for i in range(half_size_of_population * 2)]
            results.sort(key=lambda tup:tup[2], reverse=True)
            for row in results:
                data_writer.writerow(row)
            pbar.update(1)
        pbar.close()
    print('|Final Generation| Best Fitness: {}| Best Gene: {}'.format(results[0][2],
                                                                      results[0][3:]))
    return results[0][2], results[0][3:]