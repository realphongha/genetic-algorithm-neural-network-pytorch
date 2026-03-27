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
    device = next(model.parameters()).device
    chromosome = torch.tensor(chromosome).to(device)
    start = 0
    for param in model.parameters():
        param_size = param.numel()
        param.data = chromosome[start:start+param_size].view(param.shape).float()
        start += param_size


def _create_mutated(configs, nn_class, individual_class, state_dict):
    net = nn_class(configs)
    net.load_state_dict(state_dict)
    ind = individual_class(configs, nn_class, net, calc_fitness=False)
    ind.mutate()
    ind.calc_fitness()
    return ind


def _cross_mutate_fitness(p1, p2):
    res = []
    for child in p1.cross(p2):
        child.mutate()
        child.calc_fitness()
        res.append(child)
    return res


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
            self.calc_fitness()

    def load_weights(self, path, calc_fitness=False):
        network = self.network_class(self.configs)
        network.load_state_dict(torch.load(path))
        if not calc_fitness:
            print("Loaded weights from %s" % path)
        self.chromosome = network
        self.chromosome.to(self.device)
        self.chromosome.eval()
        if calc_fitness:
            self.calc_fitness()

    def save_weights(self, file_name):
        os.makedirs(self.configs["save_path"], exist_ok=True)
        torch.save(
            self.chromosome.state_dict(),
            os.path.join(self.configs["save_path"], file_name)
        )

    def random_init(self):
        self.chromosome = self.network_class(self.configs)
        self.chromosome.init_weights(self.uniform_a, self.uniform_b)

    def cross(self, other):
        child_net1 = self.network_class(self.configs)
        child_net2 = self.network_class(self.configs)
        for p1, p2, c1, c2 in zip(
            self.chromosome.parameters(), other.chromosome.parameters(),
            child_net1.parameters(), child_net2.parameters()
        ):
            mask = torch.rand(p1.shape).to(self.device) < 0.5
            c1.data = torch.where(mask, p1.data, p2.data).clone()
            c2.data = torch.where(mask, p2.data, p1.data).clone()
        yield self.__class__(self.configs, self.network_class, child_net1, calc_fitness=False)
        yield self.__class__(self.configs, self.network_class, child_net2, calc_fitness=False)

    def mutate_param(self):
        if random.random() > self.mutation_rate:
            return False
        params = random.choice(list(self.chromosome.parameters()))
        flat = params.data.view(-1)
        idx = random.randint(0, flat.shape[0] - 1)
        flat[idx] = random.uniform(self.uniform_a, self.uniform_b)
        return True

    def mutate_param_add(self):
        if random.random() > self.mutation_rate:
            return False
        params = random.choice(list(self.chromosome.parameters()))
        flat = params.data.view(-1)
        idx = random.randint(0, flat.shape[0] - 1)
        flat[idx] += np.random.normal(0, self.mutation_strength)
        return True


    def mutate_layer_add(self):
        mutated = False
        for p in self.chromosome.parameters():
            if random.random() > self.mutation_rate:
                continue
            sh = p.data.shape
            randn_mask = torch.randn(sh) * self.mutation_strength
            p.data += randn_mask.to(self.device)
            mutated = True
        return mutated

    def mutate(self):
        if self.configs["mutation_type"] == "layer_add":
            return self.mutate_layer_add()
        elif self.configs["mutation_type"] == "param":
            return self.mutate_param()
        elif self.configs["mutation_type"] == "param_add":
            return self.mutate_param_add()
        raise NotImplementedError(
            f"{self.configs['mutation_type']} not implemented!"
        )

    def __hash__(self):
        # use a faster hashing method, or just use the id of the chromosome
        return id(self.chromosome)


class GeneticAlgorithmNN(GeneticAlgorithm):
    INDIVIDUAL_CLASS = IndividualNN
    NN_CLASS = None

    def __init__(self, configs: dict, pretrained_weights: str = ''):
        super().__init__(configs)
        self.save_path = configs["save_path"]
        self.pretrained_weights = pretrained_weights \
            if os.path.isfile(pretrained_weights) else None
        if self.configs["workers"] and self.configs["workers"] > 1:
            try:
                mp.set_start_method('spawn')
            except RuntimeError:
                pass
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
            state_dict = torch.load(self.pretrained_weights)
            network = self.NN_CLASS(self.configs)
            network.load_state_dict(state_dict)
            individual = self.INDIVIDUAL_CLASS(self.configs, self.NN_CLASS, network)
            self.population.append(individual)

            num_to_create = self.configs["num_parents"] - len(self.population)
            if num_to_create > 0:
                if self.configs["workers"] and self.configs["workers"] > 1:
                    mutated_indivs = self.pool.starmap(
                        _create_mutated,
                        [(self.configs, self.NN_CLASS, self.INDIVIDUAL_CLASS, state_dict) 
                         for _ in range(num_to_create)]
                    )
                else:
                    mutated_indivs = [_create_mutated(self.configs, self.NN_CLASS, self.INDIVIDUAL_CLASS, state_dict) 
                                      for _ in range(num_to_create)]
                self.population.extend(mutated_indivs)

        self.population.extend(self.new_population(self.population_size - len(self.population)))

    def crossover_and_mutation(self, population):
        if not self.configs["workers"] or self.configs["workers"] <= 1:
            children = []
            while len(children) < self.population_size:
                parent1, parent2 = random.sample(population, 2)
                for child in parent1.cross(parent2):
                    child.mutate()
                    child.calc_fitness()
                    children.append(child)
                    if len(children) >= self.population_size:
                        break
            return children

        # Multiprocessing
        pairs = []
        while len(pairs) * 2 < self.population_size:
            pairs.append(random.sample(population, 2))

        results = self.pool.starmap(_cross_mutate_fitness, pairs)
        children = [child for sublist in results for child in sublist]
        return children[:self.population_size]

    def loop_callback(self, greatest_of_this_gen):
        if greatest_of_this_gen > self.goat:
            greatest_of_this_gen.save_weights("best.pth")
            print(f"Saved weights to {os.path.join(self.configs['save_path'], 'best.pth')}")

        greatest_of_this_gen.save_weights("last.pth")
        print(f"Saved weights to {os.path.join(self.configs['save_path'], 'last.pth')}")

