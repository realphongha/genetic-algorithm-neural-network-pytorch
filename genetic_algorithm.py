import time
from abc import ABC, abstractmethod
from typing import List

from tqdm import tqdm


class Individual(ABC):
    def __init__(self, configs, chromosome=None):
        self.configs = configs
        # chromosome encodes a solution for the problem
        self.chromosome = chromosome
        # mutation rate
        self.mutation_rate = configs["mutation_rate"]
        # mutation rate
        self.mutation_strength = configs["mutation_strength"]
        # fitness means how good this solution is
        if chromosome:
            self.fitness = self.calc_fitness()
        else:
            self.fitness = None
        self.debug = configs["debug"]

    @abstractmethod
    def display(self):
        # prints the chromosome to screen
        pass

    @abstractmethod
    def random_init(self):
        # randomly initializes an individual
        pass

    @abstractmethod
    def calc_fitness(self):
        # calculates this individual fitness
        pass

    @abstractmethod
    def cross(self, other):
        # crosses with another individual
        pass

    @abstractmethod
    def mutate(self):
        # mutates
        pass

    def is_valid(self):
        # is this a valid solution?
        # override it if needed
        return True

    def __lt__(self, other):
        # for comparing individuals
        return self.fitness < other.fitness

    def __eq__(self, other):
        # override it if needed
        return self.chromosome == other.chromosome

    @abstractmethod
    def __hash__(self):
        pass


class GeneticAlgorithm(ABC):
    def __init__(self, configs: dict):
        self.configs = configs
        # how many individuals in the population?
        self.population_size = configs["population_size"]
        # how many parents to be selected for crossover?
        self.num_parents = configs["num_parents"]
        assert self.num_parents < self.population_size
        # all individuals in current generation
        self.population: List[Individual] = list()
        # keeps best individuals from last generation or not? percentage or None
        self.elitism = configs.get("elitism", None)
        assert self.elitism is None or 0 <= self.elitism <= 1
        self.max_gen = configs.get("max_gen", None)
        self.max_gen_without_evolution = configs.get("max_gen_without_evolution", None)
        # verbose or not?
        self.debug = configs.get("debug", True)
        # greatest of all time
        self.goat: Individual = None

    @abstractmethod
    def init_population(self):
        # initializes population
        pass

    @abstractmethod
    def selection(self):
        # selects some individuals to reproduce next generation
        pass

    @abstractmethod
    def crossover(self, population: List[Individual]):
        # does crossover to produce offsprings
        pass

    @abstractmethod
    def mutation(self, population: List[Individual]):
        # mutates the population
        pass

    @abstractmethod
    def can_terminate(self, evolved: bool, gen: int):
        # can the algorithm stop now?
        pass

    def run(self) -> Individual:
        # main loop of the algorithm
        self.init_population()
        self.goat = max(self.population)
        evolved = False
        gen = 0
        running_latency = []
        while not self.can_terminate(evolved, gen):
            try:
                start = time.time()
                gen += 1
                print("Generation %i:" % gen)
                self.population.sort(reverse=True)
                parents = self.selection()
                children = self.crossover(parents)
                children = self.mutation(children)
                greatest = max(children)
                if greatest > self.goat:
                    self.goat = greatest
                    evolved = True
                else:
                    evolved = False
                print("Best individual in this generation:")
                greatest.display()
                if evolved:
                    print("Evolved!")
                if not self.elitism:
                    self.population = children
                else:
                    self.population = self.population[int(self.elitism * self.population_size):] + children
                latency = time.time() - start
                running_latency.append(latency)
                if self.debug:
                    print("Time: %.4f (s)" % latency)
                    print()
            except KeyboardInterrupt:
                print("Gracefully stop!")
                break
        print("Stop evolved!")
        print("Greatest of all time:")
        self.goat.display()
        print("Running time: %.4f (s)" % sum(running_latency))
        print("Avg. running time per gen: %.4f (s)" %
            (sum(running_latency)/len(running_latency)))

