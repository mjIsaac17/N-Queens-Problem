# 8 Queens problem. Genetic Algoriths
# By: Isaac Montes Jiménez
# Date: 9/15/2019
# Modified: 9/18/2019

import random as r
from matplotlib import pyplot as p_plot


#Constants
POPULATION_SIZE = 100
MUTATION_PROB = 0.8
MAX_FITNESS_EVALUATION = 10000
BOARD_SIZE = 8
MAX_COLLISIONS = (BOARD_SIZE - 1) * BOARD_SIZE 
FITNESS = 1
AMOUNT_PARENT_SELECTION = 5
BEST_FROM_PARENT_SELECTION = 2
#----------------------------------------------------------
#Generations = MAX_FITNESS_EVALUATIONS / POPULATION_SIZE  -
#EJ.                                                      -
#Generations = 30,000 / 100                               -
#Generations = 300                                        -
#----------------------------------------------------------

#List for average fitness of each generation
individuals_average = []

#List to save the total of generations
generations = [0]

class Individual():
    genotype = [] 
    fitness = 0

# Create a random board for an individual 
# values betweem 1 and board size
def fnRandomBoard():
    genotype = []
    for i in range(0,BOARD_SIZE):
        genotype.append(r.randrange(1,BOARD_SIZE+1))
    return genotype
    
# Function to calculate individual's fitness
def fnFitnessCalculation(colissions):
    fitness = FITNESS - (colissions / MAX_COLLISIONS)
    return fitness

def fnCheckFitness(individual):
    if(individual.fitness == FITNESS):
        print("FITNESS FOUND!!")
        print("Solution: ", individual.genotype)
        fnPlot(individuals_average)
        exit()

# Initialize the population 
def fnInitializePopulation():
    population = []
    collisions = 0
    for i in range (0, POPULATION_SIZE):
        population.append(Individual())
        population[i].genotype = fnRandomBoard()
        collisions = fnCollisions(population[i].genotype)
        
        population[i].fitness = fnFitnessCalculation(collisions)
        fnCheckFitness(population[i])
    return population

#Count individual's collisions
def fnCollisions(genotype):
    totalCollisions = 0
    for i in range(0,BOARD_SIZE):
        #count collisions in the board by row
        totalCollisions += genotype.count(genotype[i])-1 

        #count lower right diagonal collisions  
        queen = genotype[i]+1
        index = i+1        
        while(queen <= BOARD_SIZE and index < BOARD_SIZE):
            if(queen == genotype[index]):
                totalCollisions += 1
            queen += 1
            index += 1

        #count lower left diagonal collisions  
        queen = genotype[i]+1
        index = i-1        
        while(queen <= BOARD_SIZE and index >= 0):
            if(queen == genotype[index]):
                totalCollisions += 1
            queen += 1
            index -= 1
        
        #count upper right diagonal collisions 
        queen = genotype[i]-1
        index = i+1
        while(queen >= 0 and index < BOARD_SIZE): 
            if(queen == genotype[index]): 
                totalCollisions += 1
            queen -= 1
            index += 1

        #count upper left diagonal collisions 
        queen = genotype[i]-1
        index = i-1      
        while(queen >= 0 and index >= 0):
            if(queen == genotype[index]): 
                totalCollisions += 1
            queen -= 1
            index -= 1

    return totalCollisions
    

#Recombinantion: beginning after crossover point
def fnCrossover(parent1, parent2):
    #with BOARD_SIZE = 8
    child1 = Individual()
    child2 = Individual()
    crossover_point = r.randrange(1,BOARD_SIZE) #values between 1 and 7
    child1.genotype = parent1.genotype[0:crossover_point]
    child1.genotype.extend(parent2.genotype[crossover_point:BOARD_SIZE])
    child2.genotype = parent2.genotype[0:crossover_point]
    child2.genotype.extend(parent1.genotype[crossover_point:BOARD_SIZE])
    return child1,child2

def fnMutation(individual):
    rand1 = r.randrange(0,BOARD_SIZE)
    rand2 = r.randrange(0,BOARD_SIZE)
    while(rand1 == rand2):
        rand2 = r.randrange(0,BOARD_SIZE)
    individual.genotype[rand1], individual.genotype[rand2] = individual.genotype[rand2], individual.genotype[rand1]
    return individual


def fnRecombinantion(population):
    lsPopulation = list(population)
    parentSelection = []
    bestParentSelection = []
    bestIndividual = Individual()

    #sort population by fitness, lower - higher
    lsPopulation = sorted(lsPopulation, key=lambda Individual: Individual.fitness)
    bestIndividual = lsPopulation[POPULATION_SIZE-1]
    hijos = 0

    #ONE GENERATION = when we have created the same amount of children that the population size
    while(hijos < POPULATION_SIZE):
        child1 = Individual()
        child2 = Individual()

        #take random parents from population
        for i in range(0, AMOUNT_PARENT_SELECTION):
            rand_index = r.choice(range(0, lsPopulation.__len__()))
            parentSelection.append(lsPopulation[rand_index])
        parentSelection = sorted(parentSelection, key=lambda Individual: Individual.fitness)
        parentSelection.reverse()

        #take best parents 
        for i in range(0, BEST_FROM_PARENT_SELECTION):
            bestParentSelection.append(parentSelection[i])
        
        #create 2 child by crossover
        child1, child2 = fnCrossover(bestParentSelection[0], bestParentSelection[1])

        #calculate and check fitness of the children
        child1.fitness = fnFitnessCalculation(fnCollisions(child1.genotype))
        #child1.fitness = 1
        fnCheckFitness(child1)
        child1.fitness = fnFitnessCalculation(fnCollisions(child1.genotype))
        fnCheckFitness(child2)

        #mutate the children and check their fitness
        mutation_prob = r.choice(range(1,11))
        mutation_prob /= 10
        if(mutation_prob <= MUTATION_PROB):
            child1 = fnMutation(child1)
            fnCheckFitness(child1)
            child2 = fnMutation(child2)
            fnCheckFitness(child2)

        #check if the children are better to add them in the population
        if(lsPopulation[0].fitness < child1.fitness):
            lsPopulation[0] = child1
        if(lsPopulation[1].fitness < child2.fitness):
            lsPopulation[1] = child2

        #obtain the best individual's fitness by generation 
        bestIndividualPopulation = lsPopulation[lsPopulation.__len__()-1]
        if(bestIndividualPopulation.fitness > bestIndividual.fitness):
            bestIndividual = bestIndividualPopulation
        hijos += 2
        lsPopulation = sorted(lsPopulation, key=lambda Individual: Individual.fitness)
    return bestIndividual, lsPopulation

#plot the average fitness by generation
def fnPlot(y_individualsAverage):
    #x_generations = [i for i in range(0,generations)]
    generations.pop(0)
    p_plot.rcParams['toolbar'] = 'None'
    p_plot.plot(generations, y_individualsAverage)
    p_plot.xlabel('Generaciones')
    p_plot.ylabel('Fitness')
    p_plot.title('Tablero %d x %d con %d reinas' %(BOARD_SIZE, BOARD_SIZE, BOARD_SIZE))
    p_plot.show()

#calculate the average population's fitness
def fnAverageFitness(population):
    average = 0
    for i in population:
        average += i.fitness
    average /= POPULATION_SIZE
    return average

#Main function 
def fnMain():
    generations[0] = 1
    count_generations = 1
    evaluations = 0
    population = []
    best_individual = Individual()
    best_generation = 0

    population = fnInitializePopulation()
    while(evaluations < MAX_FITNESS_EVALUATION):
        print("Generation: ", count_generations)      
        best_individual_generation = Individual()
        best_individual_generation, population = fnRecombinantion(population)
        individuals_average.append(fnAverageFitness(population))
        print("Best individual in the generation: ",best_individual_generation.fitness, "\nBoard: ", best_individual_generation.genotype)
        if(best_individual.fitness < best_individual_generation.fitness):
            best_individual = best_individual_generation
            best_generation = count_generations
        evaluations += POPULATION_SIZE
        count_generations += 1
        generations.append(count_generations)
        print("_____________________________________________")
    print("Best individual: ", best_individual.fitness, "\nBoard: ", best_individual.genotype)
    print("Best individual in generation: ", best_generation)
    fnPlot(individuals_average)

# Initialize main function
fnMain()


