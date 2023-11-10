import random
from mspd import solve


class GeneticAgent:
    # Settings
    generations = 400
    population_size = 100
    crossover_rate = 0.7
    mutation_rate = 0.2
    tournament_size = 20

    def __init__(self, N, objectiveN, inputDf):
        self.N = N
        self.objectiveN = objectiveN
        self.inputDf = inputDf

        self.fitness_store = {}


    def fitness(self, chromosome):
        # Compute the fitness (score) of each chromosome as the skew value in the constructed PD tree
        if tuple(chromosome) in self.fitness_store:
            return self.fitness_store[tuple(chromosome)]

        wl, skew = solve(self.N, chromosome, self.inputDf)
        self.fitness_store[tuple(chromosome)] = -skew
        objective = 0.2 * (wl + skew) + 0.4 * (3 * wl + skew) + 0.4 * (wl + 3 * skew)
        return -objective


    def tournament_selection(self, population):
        # Select the top parent from a randomised sample
        return max(random.sample(population, GeneticAgent.tournament_size), key=self.fitness)


    def crossover(self, parent1, parent2):
        gene_length = len(parent1)

        if gene_length <= 1 or random.random() < GeneticAgent.crossover_rate:
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


    def mutate(self, chromosome):
        if random.random() < GeneticAgent.mutation_rate:
            return chromosome

        mutation_index = random.randint(0, len(chromosome) - 1)
        while True:
            new_source = random.choice(range(1, self.N))
            if new_source in chromosome:
                # Note that the length of chromosome is again at most 3.
                continue

            chromosome[mutation_index] = new_source
            break

        return chromosome


    def select_best_vertices(self):
        population = [sorted(random.sample(range(1, self.N), self.objectiveN)) for i in range(GeneticAgent.population_size)]
        best_chromosomes = []

        for generation in range(GeneticAgent.generations):
            # Populate the next generation
            selected_parents = [self.tournament_selection(population) for _ in range(GeneticAgent.population_size)]
            next_generation = []

            for i in range(0, GeneticAgent.population_size, 2):
                parent1, parent2 = selected_parents[i], selected_parents[i + 1]
                for child in self.crossover(parent1, parent2):
                    next_generation.append(sorted(self.mutate(child)))

            population = next_generation
            best_chromosomes.append(max(population, key=self.fitness))

        return max(best_chromosomes, key=self.fitness)



# import pandas as pd
#
# inputDf = pd.read_csv("testcases/input_stt_45.csv.gz", compression="gzip")
#
#
# genetic_agent = GeneticAgent(45, 2, inputDf[inputDf["netIdx"] == 299])
# print(genetic_agent.select_best_vertices())
