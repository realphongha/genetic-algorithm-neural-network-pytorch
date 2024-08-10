import random
import os

import torch
import numpy as np

from genetic_algorithm import Individual, GeneticAlgorithm


def model_to_chromosome(model):
    chromosome = []
    for params in model.parameters():
        chromosome.append(params.data.cpu().numpy().flatten())
    return torch.tensor(np.concatenate(chromosome))


def chromosome_to_model(chromosome, model):
    chromosome = torch.tensor(chromosome).cuda()
    start = 0
    for param in model.parameters():
        param_size = param.numel()
        param.data = chromosome[start:start+param_size].view(param.shape).float()
        start += param_size


class IndividualNN(Individual):
    def __init__(self, configs, network_class, network=None):
        super().__init__(configs)
        self.network_class = network_class
        self.device = configs["device"]
        self.mean = 0.0
        self.std = 1.0
        if network is None:
            self.random_init()
        else:
            self.network = network
        self.network.to(self.device)
        self.network.eval()
        self.chromosome = model_to_chromosome(self.network)
        self.fitness = self.calc_fitness()

    def load_weights(self, path):
        network = self.network_class(self.configs)
        network.load_state_dict(torch.load(path))
        print("Loaded weights from %s" % path)
        self.network = network
        self.network.to(self.device)
        self.network.eval()
        self.chromosome = model_to_chromosome(self.network)
        self.fitness = self.calc_fitness()

    def random_init(self):
        self.network = self.network_class(self.configs)
        self.network.init_weights(self.mean, self.std)

    def cross(self, other):
        cross_idx = random.randint(0, len(self.chromosome) - 1)

        child1 = np.concatenate(
            (self.chromosome[:cross_idx], other.chromosome[cross_idx:]),
            axis=0
        )
        child1_net = self.network_class(self.configs)
        chromosome_to_model(child1, child1_net)
        yield self.__class__(self.configs, self.network_class, child1_net)

        child2 = np.concatenate(
            (other.chromosome[:cross_idx], self.chromosome[cross_idx:]),
            axis=0
        )
        child2_net = self.network_class(self.configs)
        chromosome_to_model(child2, child2_net)
        yield self.__class__(self.configs, self.network_class, child2_net)

    def mutate(self):
        if random.random() < self.mutation_rate:
            cross_idx = random.randint(0, len(self.chromosome) - 1)
            self.chromosome[cross_idx] = random.gauss(self.mean, self.std)
            chromosome_to_model(self.chromosome, self.network)
            self.fitness = self.calc_fitness()

    def __hash__(self):
        return hash(self.chromosome)


class GeneticAlgorithmNN(GeneticAlgorithm):
    def __init__(self, configs: dict):
        super().__init__(configs)
        self.save_path = configs["save_path"]

    def init_population(self):
        for _ in range(self.population_size):
            self.population.append(IndividualNN(self.configs, self.network_class))

    def run(self):
        super().run()
        if self.save_path is not None:
            save_dir = os.path.split(self.save_path)[0]
            os.makedirs(save_dir, exist_ok=True)
            torch.save(self.goat.network.state_dict(), self.save_path)
            print("Saved the GOAT to %s" % self.save_path)

