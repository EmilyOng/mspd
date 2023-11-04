import random
from mspd import solve


def genetic_algorithm(N, objectiveN, inputDf):
    # Settings
    generations = 100
    population_size = 20
    crossover_rate = 0.7
    mutation_rate = 0.4
    tournament_size = 5

    fitness_store = {}

    def fitness(chromosome):
        # Compute the fitness (score) of each chromosome as the skew value in the constructed PD tree
        if tuple(chromosome) in fitness_store:
            return fitness_store[tuple(chromosome)]

        wl, skew = solve(N, chromosome, inputDf)
        fitness_store[tuple(chromosome)] = -skew
        return -skew


    def tournament_selection(population):
        # Select the top parent from a randomised sample
        return max(random.sample(population, tournament_size), key=fitness)


    def crossover(parent1, parent2):
        gene_length = len(parent1)

        if gene_length <= 1 or random.random() < crossover_rate:
            return parent1, parent2

        # Create a child chromosome by combining genes from two parent chromosomes
        # while preserving a middle section of genes from the first parent. This is required
        # instead of the standard crossover operation due to possibility of creating invalid
        # genes with duplicate values.
        # https://github.com/giacomelli/GeneticSharp/blob/master/src/GeneticSharp.Domain/Crossovers/OrderedCrossover.cs

        middle_section_begin, middle_section_end = sorted(random.sample(range(gene_length), 2))

        # Create the first child
        child1 = []
        # This is O(1) since the gene length is at most 3.
        parent1_middle_genes = parent1[middle_section_begin:middle_section_end + 1]
        parent2_remaining_genes = iter([gene for gene in parent2 if gene not in parent1_middle_genes])

        for i in range(gene_length):
            if middle_section_begin <= i <= middle_section_end:
                child1.append(parent1[i])
            else:
                child1.append(next(parent2_remaining_genes))


        # Create the second child
        child2 = []
        parent2_middle_genes = parent2[middle_section_begin:middle_section_end + 1]
        parent1_remaining_genes = iter([gene for gene in parent1 if gene not in parent2_middle_genes])

        for i in range(gene_length):
            if middle_section_begin <= i <= middle_section_end:
                child2.append(parent2[i])
            else:
                child2.append(next(parent1_remaining_genes))

        return child1, child2


    def mutate(chromosome):
        if random.random() < mutation_rate:
            return chromosome

        mutation_index = random.randint(0, len(chromosome) - 1)
        while True:
            new_source = random.choice(range(1, N))
            if new_source in chromosome:
                # Note that the length of chromosome is again at most 3.
                continue

            chromosome[mutation_index] = new_source
            break

        return chromosome


    population = [sorted(random.sample(range(1, N), objectiveN)) for i in range(population_size)]
    best_chromosomes = []

    for generation in range(generations):
        # Populate the next generation
        selected_parents = [tournament_selection(population) for _ in range(population_size)]
        next_generation = []

        for i in range(0, population_size, 2):
            parent1, parent2 = selected_parents[i], selected_parents[i + 1]
            for child in crossover(parent1, parent2):
                next_generation.append(sorted(mutate(child)))

        population = next_generation
        best_chromosomes.append(max(population, key=fitness))

    return max(best_chromosomes, key=fitness)
