import random

import torch

from genetic_algorithm_neural_network import GeneticAlgorithmNN, IndividualNN
from xor_calculation.xor_nn import XorNN


X_TRAIN = torch.tensor([
    [0, 0],  # 0 xor 0 == 0
    [0, 1],  # 0 xor 1 == 1
    [1, 0],  # 1 xor 0 == 1
    [1, 1],  # 1 xor 1 == 0
]).float()


class XorIndividualNN(IndividualNN):
    def __init__(self, configs, network_class, network=None):
        super().__init__(configs, network_class, network)

    def display(self):
        res = self.network(X_TRAIN.to(self.device)).cpu().detach().numpy()
        print("0 xor 0 =", res[0])
        print("0 xor 1 =", res[1])
        print("1 xor 0 =", res[2])
        print("1 xor 1 =", res[3])

    def calc_fitness(self):
        conf = 0.0
        res = self.network(X_TRAIN.to(self.device)).cpu().detach().numpy()
        conf += res[0][0].item()
        conf += res[1][1].item()
        conf += res[2][1].item()
        conf += res[3][0].item()
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
    configs = {
        "population_size": 100,
        "num_parents": 50,
        "mutation_rate": 0.02,
        "elitism": 0.1,
        "max_gen": 100,
        "device": "cuda",
        "save_path": "./weights/xor_calc.pth",
        "debug": True
    }
    xor = XorCalculationGANN(configs)
    xor.run()
    goat = XorIndividualNN(configs, XorNN)
    goat.load_weights("./weights/xor_calc.pth")
    goat.display()

