import random

import torch

from genetic_algorithm_neural_network import GeneticAlgorithmNN, IndividualNN
from xor_calculation.xor_nn import XorNN


ZERO_ZERO = torch.tensor([0, 0]).float()
ZERO_ONE = torch.tensor([0, 1]).float()
ONE_ZERO = torch.tensor([1, 0]).float()
ONE_ONE = torch.tensor([1, 1]).float()


class XorIndividualNN(IndividualNN):
    def __init__(self, configs, network_class, network=None):
        super().__init__(configs, network_class, network)

    def display(self):
        print("0 xor 0 =", self.network(ZERO_ZERO.to(self.device)).cpu().detach().numpy())
        print("0 xor 1 =", self.network(ZERO_ONE.to(self.device)).cpu().detach().numpy())
        print("1 xor 0 =", self.network(ONE_ZERO.to(self.device)).cpu().detach().numpy())
        print("1 xor 1 =", self.network(ONE_ONE.to(self.device)).cpu().detach().numpy())

    def calc_fitness(self):
        conf = 0.0
        conf += self.network(ZERO_ZERO.to(self.device))[0].item()
        conf += self.network(ZERO_ONE.to(self.device))[1].item()
        conf += self.network(ONE_ZERO.to(self.device))[1].item()
        conf += self.network(ONE_ONE.to(self.device))[0].item()
        return conf


class XorCalculationGANN(GeneticAlgorithmNN):
    def __init__(self, configs: dict):
        super().__init__(configs)

    def init_population(self):
        for _ in range(self.population_size):
            self.population.append(XorIndividualNN(self.configs, XorNN))

    def selection(self):
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        parents = self.population[:self.num_parents]
        return parents

    def crossover(self, population):
        children = []
        while True:
            parent1 = population[random.randint(0, len(population) - 1)]
            parent2 = population[random.randint(0, len(population) - 1)]
            for child in parent1.cross(parent2):
                children.append(child)
                if len(children) >= self.population_size:
                    break
            if len(children) >= self.population_size:
                break
        return children

    def mutation(self, population):
        for individual in population:
            individual.mutate()
        return population

    def can_terminate(self, evolved, gen):
        return gen >= self.max_gen


if __name__ == "__main__":
    xor = XorCalculationGANN(configs={
        "population_size": 100,
        "num_parents": 50,
        "mutation_rate": 0.02,
        "elitism": 0.1,
        "max_gen": 100,
        "device": "cuda",
        "debug": True
    })
    xor.run()
