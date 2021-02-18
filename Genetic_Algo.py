import random as rn
import numpy as np
from numpy.random import choice
import math
import matplotlib.pyplot as plt
rn.seed(13)

INF = 1001


# the adjacency matrix denoting the distance between the cities
adj = [
    [0, INF, INF, INF, INF, INF, .15,
        INF, INF, .2, INF, .12, INF, INF],
    [INF, 0, INF, INF, INF, INF,
        INF, .19, .4, INF, INF, INF, INF, .13],
    [INF, INF, 0, .6, .22, .4, INF, INF, .2,
        INF, INF, INF, INF, INF],
    [INF, INF, .6, 0, INF, .21, INF, INF,
        INF, INF, .3, INF, INF, INF],
    [INF, INF, .22, INF, 0, INF, INF,
        INF, .18, INF, INF, INF, INF, INF],
    [INF, INF, .4, .21, INF, 0, INF,
        INF, INF, INF, .37, .6, .26, .9],
    [.15, INF, INF, INF, INF, INF, 0,
        INF, INF, INF, .55, .18, INF, INF],
    [INF, .19, INF, INF, INF, INF, INF,
        0, INF, .56, INF, INF, INF, .17],
    [INF, .4, .2, INF, .18, INF, INF,
        INF, 0, INF, INF, INF, INF, .6],
    [.2, INF, INF, INF, INF, INF,
        INF, .56, INF, 0, INF, .16, INF, .5],
    [INF, INF, INF, .3, INF, .37, .55,
        INF, INF, INF, 0, INF, .24, INF],
    [.12, INF, INF, INF, INF, .6, .18,
        INF, INF, .16, INF, 0, .4, INF],
    [INF, INF, INF, INF, INF, .26, INF,
        INF, INF, INF, .24, .4, 0, INF],
    [INF, .13, INF, INF, INF, .9,
        INF, .17, .6, .5, INF, INF, INF, 0]
]


def path_cost(state):
    n = len(state)
    cost = 0
    for i in range(n-1):
        cost += (adj[ord(state[i])-65][ord(state[i+1])-65])
    cost += adj[ord(state[n-1])-65][ord(state[0])-65]
    return round(cost, 2)


def calc_fitness(population):
    fitness = []
    for state in population:
        fitness.append((1/path_cost(state)))
    return fitness


def calc_fitness_state(state):
    return (1/path_cost(state))


def random_selecion(population, fitness, number=2):
    n = len(fitness)
    sum_of_fitness = np.sum(fitness)
    probabilities = [fitness[i]/sum_of_fitness for i in range(n)]
    draw = choice(population, number, p=probabilities, replace=False)
    return draw


def mutate(state):
    n = len(state)
    idx1 = rn.randint(0, n-1)
    idx2 = rn.randint(0, n-1)
    while(idx1 == idx2):
        idx2 = rn.randint(0, n-1)
    mutated = ""
    for i in range(n):
        if(i == idx1):
            mutated += state[idx2]
        elif(i == idx2):
            mutated += state[idx1]
        else:
            mutated += state[i]
    return mutated


def reproduce(par1, par2):
    n = len(par1)
    idx1 = rn.randint(0, n-1)
    idx2 = rn.randint(idx1, n-1)
    child = par2[idx1:idx2+1]
    replace = {}
    for i in range(idx1, idx2+1):
        replace[par2[i]] = 1

    i = 0
    count = 0
    temp = ""
    while(count < idx1):
        if par1[i] not in replace.keys():
            temp += par1[i]
            count += 1
        i += 1

    child = temp+child
    count = idx2+1
    while(count < n):
        if par1[i] not in replace.keys():
            child += par1[i]
            count += 1
        i += 1

    return child


def genetic_algo(population):
    new_population = []
    n = len(population)
    fitness = calc_fitness(population)
    for _ in range(n):
        x, y = random_selecion(population, fitness)
        z = reproduce(x, y)
        if(rn.random() > 0.8):
            z = mutate(z)
        new_population.append(z)
    return new_population


def random_selecion_improved(population, fitness, number=2):
    n = len(fitness)
    temp = [(f+1)**2 for f in fitness]
    sum_of_fitness = np.sum(temp)
    probabilities = [temp[i]/sum_of_fitness for i in range(n)]
    draw = choice(population, number, p=probabilities, replace=False)
    return draw


def reproduce_improved(par1, par2):
    n = len(par1)
    subs = rn.randint(0, n-1)
    if calc_fitness_state(par1) > calc_fitness_state(par2):
        max_fitness = calc_fitness_state(par1)
        best_child = par1
    else:
        max_fitness = calc_fitness_state(par2)
        best_child = par2

    for idx1 in range(n-subs):
        idx2 = idx1+subs
        child = par2[idx1:idx2+1]
        replace = {}
        for i in range(idx1, idx2+1):
            replace[par2[i]] = 1

        i = 0
        count = 0
        temp = ""
        while(count < idx1):
            if par1[i] not in replace.keys():
                temp += par1[i]
                count += 1
            i += 1

        child = temp+child
        count = idx2+1
        while(count < n):
            if par1[i] not in replace.keys():
                child += par1[i]
                count += 1
            i += 1
        if calc_fitness_state(child) > max_fitness:
            max_fitness = calc_fitness_state(child)
            best_child = child

    for idx1 in range(n-subs):
        idx2 = idx1+subs
        child = par1[idx1:idx2+1]
        replace = {}
        for i in range(idx1, idx2+1):
            replace[par1[i]] = 1

        i = 0
        count = 0
        temp = ""
        while(count < idx1):
            if par2[i] not in replace.keys():
                temp += par2[i]
                count += 1
            i += 1

        child = temp+child
        count = idx2+1
        while(count < n):
            if par2[i] not in replace.keys():
                child += par2[i]
                count += 1
            i += 1
        if calc_fitness_state(child) > max_fitness:
            max_fitness = calc_fitness_state(child)
            best_child = child

    return best_child


def mutate_improved(state):
    n = len(state)
    neighbours = [state]
    for idx1 in range(n):
        for idx2 in range(idx1+1, n):
            mutated = ""
            for i in range(n):
                if(i == idx1):
                    mutated += state[idx2]
                elif(i == idx2):
                    mutated += state[idx1]
                else:
                    mutated += state[i]
            neighbours.append(mutated)

    fitness = calc_fitness(neighbours)
    new_child = neighbours[fitness.index(max(fitness))]
    while new_child == state:
        idx = rn.randint(0, len(neighbours)-1)
        new_child = neighbours[idx]

    return new_child


def genetic_algo_improved(population):
    new_population = []
    n = len(population)
    fitness = calc_fitness(population)
    for _ in range(n):
        x, y = random_selecion_improved(population, fitness)
        z = reproduce_improved(x, y)
    # this part uses the double mutation to reduce the possibility of getting stuck into a local maximum
        if (rn.random() > 0.4):
            z = mutate_improved(z)
        new_population.append(z)

    # take the newly formed population along with the parent population to find the best candidates for the next iterations
    # temp_population = new_population+population
    # temp_fitness = calc_fitness(temp_population)
    # temp_population = [x for _, x in sorted(
    #     zip(temp_fitness, temp_population), reverse=True)]
    # return temp_population[0:n]
    return new_population


def main():
    # -------------------------------- SECTION FOR AVERAGE PERFORMANCE -------------------------------------------------------

    # Uncomment and run the following code for getting the statistics of average performance of both the algorithms over several epochs

    # set the max generations allowed and the number of epochs
    # num_epochs=10
    # max_gen=5000

    # print('\n\n------------------------------------------ TEXTBOOK VERSION -------------------------------------------------\n\n')

    # gen_sum = 0
    # gen_not_conv = 0
    # gen_conv = []
    # for epoch in range(num_epochs):
    #     state = "ABCDEFGHIJKLMN"
    #     population = []
    #     overall_maximum = 0
    #     overall_maximum_state = ""
    #     gen_for_best_result = max_gen
    #     for i in range(20):
    #         population.append(state)

    #     for gen in range(max_gen):
    #         # print("Generation: ", (gen+1), end='\t')

    #         population = genetic_algo(population)
    #         fitness = calc_fitness(population)
    #         max_fitness = 0
    #         state_with_max_fitness = []
    #         for i in range(20):
    #             if(fitness[i] > max_fitness):
    #                 max_fitness = fitness[i]
    #                 state_with_max_fitness = population[i]

    #         if(max_fitness > overall_maximum):
    #             overall_maximum = max_fitness
    #             overall_maximum_state = state_with_max_fitness
    #             gen_for_best_result = gen+1

    #         # print("Max Fitness: ", max_fitness,
    #         #       "state with max fitness: ", state_with_max_fitness)

    #     print("Final Results for epoch", epoch+1, " :- ", end='\t')
    #     if path_cost(overall_maximum_state)>INF:
    #       print("Did not converge ")
    #       gen_not_conv+=1
    #       print("Best possible Results:- Generations: ", gen_for_best_result ,"overall maximum fitness: ",
    #             overall_maximum, "State: ", overall_maximum_state, "Path cost:- ", path_cost(overall_maximum_state)*1000)
    #       continue
    #     gen_conv.append(gen_for_best_result)
    #     print("Generations:- ", gen_for_best_result, end='\t')
    #     print("Overall maximum fitness: ", overall_maximum, end='\t')
    #     print("Path: ", overall_maximum_state, end='\t')
    #     print("Minimum cost path: ", path_cost(overall_maximum_state)*1000)

    # print("Total epochs:- ", num_epochs)
    # print("Number of epochs without convergence:- ", gen_not_conv)
    # if num_epochs>gen_not_conv:
    #   print("Average Number of Generations for Convergence:- ", np.sum(gen_conv)/(num_epochs-gen_not_conv))

    # print('\n\n------------------------------------------ IMPROVED VERSION -------------------------------------------------\n\n')

    # gen_sum = 0
    # gen_not_conv = 0
    # gen_conv = []
    # for epoch in range(num_epochs):
    #     state = "ABCDEFGHIJKLMN"
    #     population = []
    #     overall_maximum = 0
    #     overall_maximum_state =""
    #     gen_for_best_result = max_gen
    #     for i in range(20):
    #         population.append(state)

    #     for gen in range(max_gen):
    #         # print("Generation: ", (gen+1), end='\t')

    #         population = genetic_algo_improved(population)
    #         fitness = calc_fitness(population)
    #         max_fitness = 0
    #         state_with_max_fitness = ""
    #         for i in range(20):
    #             if(fitness[i] > max_fitness):
    #                 max_fitness = fitness[i]
    #                 state_with_max_fitness = population[i]

    #         if(max_fitness > overall_maximum):
    #             overall_maximum = max_fitness
    #             overall_maximum_state = state_with_max_fitness
    #             gen_for_best_result = gen+1

    #         # print("Max Fitness: ", max_fitness,
    #         #       "state with max fitness: ", state_with_max_fitness)

    #     print("Final Results for epoch", epoch+1, " :- ", end='\t')
    #     if path_cost(overall_maximum_state)>INF:
    #       print("Did not converge ")
    #       gen_not_conv+=1
    #       print("Best possible Results:- Generations: ", gen_for_best_result ,"overall maximum fitness: ",
    #             overall_maximum, "State: ", overall_maximum_state, "Path cost:- ", path_cost(overall_maximum_state)*1000)
    #       continue
    #     gen_conv.append(gen_for_best_result)
    #     print("Generations:- ", gen_for_best_result, end='\t')
    #     print("Overall maximum fitness: ", overall_maximum, end='\t')
    #     print("Path: ", overall_maximum_state, end='\t')
    #     print("Minimum cost path: ", path_cost(overall_maximum_state)*1000)

    # print("Total epochs:- ", num_epochs)
    # print("Number of epochs without convergence:- ", gen_not_conv)
    # if num_epochs>gen_not_conv:
    #   print("Average Number of Generations for Convergence:- ", np.sum(gen_conv)/(num_epochs-gen_not_conv))

    # UNCOMMENT TILL THE LINE ABOVE

    # --------------------------------XXX--------------------------------XXX--------------------------------XXX--------------------------------
    # ----------------------------------------------------------- END OF AVERAGE VERSION ------------------------------------------------------
    # --------------------------------XXX--------------------------------XXX--------------------------------XXX--------------------------------

    # ---------------------------------------------- SECTION FOR SINGLE PERFORMANCE AND GRAPH PLOT --------------------------------------------

    # Set max_gen for specifying the maximum number of generations before stopping
    max_gen = 2500

    print('------------------------------------------ TEXTBOOK VERSION -------------------------------------------------', end='\n\n')

    GA_fitness = []
    state = "ABCDEFGHIJKLMN"
    population = []
    overall_max_fitness = 0
    state_with_overall_max = 'ABCDEFGHIJKLMN'
    gen_for_max_fitness = 1
    for i in range(20):
        population.append(state)

    for gen in range(max_gen):
        print("Generation: ", (gen+1), "( Textbook Version )")

        population = genetic_algo(population)
        fitness = calc_fitness(population)
        max_fitness = 0
        state_with_max_fitness = []
        for i in range(20):
            if (fitness[i] > max_fitness):
                max_fitness = fitness[i]
                state_with_max_fitness = population[i]

        GA_fitness.append(max_fitness)
        print("Max Fitness: ", max_fitness,
              "\tstate with max fitness: ", state_with_max_fitness, "\tPath cost: ", path_cost(state_with_max_fitness)*1000)

        if max_fitness > overall_max_fitness:
            overall_max_fitness = max_fitness
            state_with_overall_max = state_with_max_fitness
            gen_for_max_fitness = gen+1

        print("overall maximum fitness: ", overall_max_fitness, "\tstate with overall maximum fitness: ", state_with_overall_max,
              "\tPath cost: ", path_cost(state_with_overall_max)*1000, end='\n\n')

    print('------------------------------------------ IMPROVED VERSION -------------------------------------------------', end='\n\n')

    GA_fitness_improved = []
    state = "ABCDEFGHIJKLMN"
    population = []
    overall_max_fitness_improved = 0
    state_with_overall_max_improved = 'ABCDEFGHIJKLMN'
    gen_for_max_fitness_improved = 1
    for i in range(20):
        population.append(state)

    for gen_improved in range(max_gen):
        print("Generation: ", (gen_improved+1), "( Improved Version )")

        population = genetic_algo_improved(population)
        fitness = calc_fitness(population)
        max_fitness_improved = 0
        state_with_max_fitness_improved = []
        for i in range(20):
            if (fitness[i] > max_fitness_improved):
                max_fitness_improved = fitness[i]
                state_with_max_fitness_improved = population[i]

        GA_fitness_improved.append(max_fitness_improved)

        print("Max Fitness: ", max_fitness_improved,
              "\tstate with max fitness: ", state_with_max_fitness_improved, "\tPath cost: ", path_cost(state_with_max_fitness_improved)*1000)

        if max_fitness_improved > overall_max_fitness_improved:
            overall_max_fitness_improved = max_fitness_improved
            state_with_overall_max_improved = state_with_max_fitness_improved
            gen_for_max_fitness_improved = gen_improved+1

        print("overall maximum fitness: ", overall_max_fitness_improved, "\tstate with overall maximum fitness: ", state_with_overall_max_improved,
              "\tPath cost: ", path_cost(state_with_overall_max_improved)*1000, end='\n\n')

    print("Final Results:- ")
    print("For textbook version, Statistics for best found results:- ")
    print("Generation: ", (gen_for_max_fitness+1), "\tMax Fitness: ", overall_max_fitness,
          "\tstate with max fitness: ", state_with_overall_max)

    if path_cost(state_with_overall_max) > INF:
        print("No path found!!")
    else:
        print('Minimum path cost found: ',
              path_cost(state_with_overall_max)*1000)

    print('\n\n')

    print("For improved version, Statistics for best found results:- ")
    print("Generation: ", (gen_for_max_fitness_improved+1), "\tMax Fitness: ", overall_max_fitness_improved,
          "\tstate with max fitness: ", state_with_overall_max_improved)

    if path_cost(state_with_overall_max_improved) > INF:
        print("No path found!!")
    else:
        print('Minimum path cost found: ', path_cost(
            state_with_overall_max_improved)*1000)

    print('\n\n')

    plt.plot(GA_fitness, label="Textbook Version", linewidth=2,)
    plt.plot(GA_fitness_improved, label="Improved Version",
             linewidth=2, )
    plt.ylabel('Best Fitness value in the population')
    plt.xlabel('Number of generations')
    plt.legend()
    plt.title(
        "Comparison of Textbook version and Improved Version for Travelling Salesman Problem")
    plt.show()

    print('\n\n')


if __name__ == '__main__':
    main()
