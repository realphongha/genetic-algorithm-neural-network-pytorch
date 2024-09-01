import random
import os

import torch
torch.set_grad_enabled(False)
import torch.multiprocessing as mp
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
    def __init__(self, configs, network_class, network=None, calc_fitness=True):
        super().__init__(configs)
        self.network_class = network_class
        self.device = configs["device"]
        self.uniform_a = -1.0
        self.uniform_b = 1.0
        if network is None:
            self.random_init()
        else:
            self.chromosome = network
        self.chromosome.to(self.device)
        self.chromosome.eval()
        if calc_fitness:
            self.fitness = self.calc_fitness()

    def load_weights(self, path, calc_fitness=False):
        network = self.network_class(self.configs)
        network.load_state_dict(torch.load(path))
        if not calc_fitness:
            print("Loaded weights from %s" % path)
        self.chromosome = network
        self.chromosome.to(self.device)
        self.chromosome.eval()
        if calc_fitness:
            self.fitness = self.calc_fitness()

    def random_init(self):
        self.chromosome = self.network_class(self.configs)
        self.chromosome.init_weights(self.uniform_a, self.uniform_b)

    def cross(self, other):
        child_net = self.network_class(self.configs)
        for p1, p2, c in zip(
            self.chromosome.parameters(), other.chromosome.parameters(), child_net.parameters()
        ):
            if random.random() < 0.5:
                c.data = p1.data.clone()
            else:
                c.data = p2.data.clone()
        yield self.__class__(self.configs, self.network_class, child_net, calc_fitness=False)

    def mutate_param(self):
        if random.random() > self.mutation_rate:
            return
        params = random.choice(list(self.chromosome.parameters()))
        flat = params.data.view(-1)
        idx = random.randint(0, flat.shape[0] - 1)
        flat[idx] = random.uniform(self.uniform_a, self.uniform_b)


    def mutate_layer(self):
        for p in self.chromosome.parameters():
            if random.random() > self.mutation_rate:
                continue
            sh = p.data.shape
            randn_mask = torch.randn(sh) * self.mutation_strength
            p.data += randn_mask.to(self.device)

    def mutate(self):
        if self.configs["mutation_type"] == "layer":
            self.mutate_layer()
        elif self.configs["mutation_type"] == "param":
            self.mutate_param()
        else:
            raise NotImplementedError(
                f"{self.configs['mutation_type']} not implemented!"
            )

    def save_weights(self, file_name):
        if not os.path.isdir(self.configs["save_path"]):
            return
        os.makedirs(self.configs["save_path"], exist_ok=True)
        torch.save(
            self.chromosome.state_dict(),
            os.path.join(self.configs["save_path"], file_name)
        )

    def __hash__(self):
        return hash(model_to_chromosome(self.chromosome))


class GeneticAlgorithmNN(GeneticAlgorithm):
    INDIVIDUAL_CLASS = IndividualNN
    NN_CLASS = None

    def __init__(self, configs: dict, pretrained_weights: str = ''):
        super().__init__(configs)
        self.save_path = configs["save_path"]
        self.pretrained_weights = pretrained_weights \
            if os.path.isfile(pretrained_weights) else None
        if self.configs["workers"] and self.configs["workers"] > 1:
            self.pool = mp.Pool(self.configs["workers"])

    def new_population(self, num):
        if self.configs["workers"] and self.configs["workers"] > 1:
            # multiprocessing for initializing new population
            return self.pool.starmap(
                self.INDIVIDUAL_CLASS,
                [(self.configs, self.NN_CLASS) for _ in range(num)]
            )
        # no multiprocessing
        return [self.INDIVIDUAL_CLASS(self.configs, self.NN_CLASS) for _ in range(num)]

    def init_population(self):
        if self.pretrained_weights:
            # if load pretrained weights
            old_mutation_rate = self.configs["mutation_rate"]
            self.configs["mutation_rate"] = 1.0
            state_dict = torch.load(self.pretrained_weights)
            network = self.NN_CLASS(self.configs)
            network.load_state_dict(state_dict)
            individual = self.INDIVIDUAL_CLASS(self.configs, self.NN_CLASS, network)
            self.population.append(individual)
            while len(self.population) < self.configs["num_parents"]:
                network = self.NN_CLASS(self.configs)
                network.load_state_dict(state_dict)
                individual = self.INDIVIDUAL_CLASS(self.configs, self.NN_CLASS, network)
                individual.mutate()
                individual.fitness = individual.calc_fitness()
                self.population.append(individual)
            self.configs["mutation_rate"] = old_mutation_rate

        self.population.extend(self.new_population(self.population_size - len(self.population)))

    def selection(self):
        parents = self.population[:self.num_parents]
        return parents

    def crossover_and_mutation(self, population):
        children = []
        while True:
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            for child in parent1.cross(parent2):
                child.mutate()
                # calculate fitness once for both cross and mutate, should be faster
                child.fitness = child.calc_fitness()
                children.append(child)
                if len(children) >= self.population_size:
                    break
            if len(children) >= self.population_size:
                break
        return children

    def loop_callback(self, greatest_of_this_gen):
        if greatest_of_this_gen > self.goat:
            greatest_of_this_gen.save_weights("best.pth")
            print("Saved weights to best.pth")
        greatest_of_this_gen.save_weights("last.pth")
        print("Saved weights to last.pth")

