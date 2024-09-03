import time
import random
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
            self.calc_fitness()
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
        # calculates this individual fitness and set to self.fitness
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
    def new_population(self, num):
        # quickly creates new population of `num` individuals
        pass

    @abstractmethod
    def init_population(self):
        # initializes population
        pass

    def selection(self):
        # selects some individuals to reproduce next generation
        if self.configs["selection_method"] == "best":
            return self.population[:self.num_parents]
        elif self.configs["selection_method"] == "best_worst":
            num_best = int(self.num_parents / 2)
            num_worst = self.num_parents - num_best
            return self.population[:num_best] + self.population[-num_worst:]
        elif self.configs["selection_method"] == "best_mid_worst":
            num_best = int(self.num_parents / 3)
            num_mid = int(self.num_parents / 3)
            num_worst = self.num_parents - num_best - num_mid
            mid_start_idx = int(len(self.population)/2 - num_mid/2)
            return self.population[:num_best] + \
                self.population[mid_start_idx:mid_start_idx+num_mid] + \
                self.population[-num_worst:]
        elif self.configs["selection_method"] == "tournament":
            new_pop = []
            while len(new_pop) < self.num_parents:
                candidates = random.sample(self.population, self.configs["k_tournament"])
                new_pop.append(max(candidates))
            return new_pop
        elif self.configs["selection_method"] == "random":
            return random.sample(self.population, self.num_parents)
        else:
            raise NotImplementedError

    @abstractmethod
    def crossover_and_mutation(self, parents: List[Individual]):
        # does crossover to produce offsprings, then mutates the offsprings
        pass

    @abstractmethod
    def can_terminate(self, evolved: bool, gen: int):
        # can the algorithm stop now?
        pass

    @abstractmethod
    def loop_callback(self, greatest_of_this_gen):
        # callback for each loop
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
                self.population.extend(self.new_population(self.configs["new_population"]))
                self.population.sort(reverse=True)
                parents = self.selection()
                children = self.crossover_and_mutation(parents)
                greatest = max(children)
                self.loop_callback(greatest)
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
                    self.population = self.population[:self.population_size]
                    self.population = self.population[int(self.elitism * self.population_size):] + children
                latency = time.time() - start
                running_latency.append(latency)
                print("Time: %.4f (s)" % latency)
                print()
            except KeyboardInterrupt:
                print("Gracefully stopped!")
                break
        print("Stop evolved!")
        print("Greatest of all time:")
        self.goat.display()
        print("Running time: %.4f (s)" % sum(running_latency))
        print("Avg. running time per gen: %.4f (s)" %
            (sum(running_latency)/len(running_latency)))

