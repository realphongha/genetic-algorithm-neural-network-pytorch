import torch
import numpy as np

from genetic_algorithm_neural_network import GeneticAlgorithmNN, IndividualNN
from snake.snake_nn import SnakeNN
from snake.snake import SnakePlayer, SnakeGame


class SnakeIndividualNN(IndividualNN):
    TURN_LEFT = {
        (1, 0): (0, -1),
        (0, -1): (-1, 0),
        (-1, 0): (0, 1),
        (0, 1): (1, 0),
    }
    TURN_RIGHT = {
        (1, 0): (0, 1),
        (0, 1): (-1, 0),
        (-1, 0): (0, -1),
        (0, -1): (1, 0),
    }

    def __init__(self, configs, network_class, network=None, calc_fitness=True):
        # how many times to run simulation
        self.simulation_times = configs["simulation_times"]
        super().__init__(configs, network_class, network, calc_fitness)

    @staticmethod
    def encode_current_position(game):
        head = game.snake[-1]
        food = game.food
        # TODO: add turn points in snake
        inp = [
            head[0]-food[0], head[1]-food[1] # food
        ]
        # is there any danger on 4 adjacent cells
        # pop the tail because the tail will move out in the next turn
        tail = game.snake.popleft()
        inp.append(head[0] == 0 or ((head[0]-1, head[1]) in game.snake))
        inp.append(head[0] == game.w-1 or ((head[0]+1, head[1]) in game.snake))
        inp.append(head[1] == 0 or ((head[0], head[1]-1) in game.snake))
        inp.append(head[1] == game.h-1 or ((head[0], head[1]+1) in game.snake))
        inp = torch.tensor(inp).float()
        game.snake.appendleft(tail)
        return inp

    @torch.no_grad()
    def get_action(self, game):
        prob = self.chromosome(SnakeIndividualNN.encode_current_position(game).to(self.device))
        direction = prob.argmax(0).item()
        if direction == 0:
            return (-1, 0)
        elif direction == 1:
            return (1, 0)
        elif direction == 2:
            return (0, -1)
        else:
            return (0, 1)

    def display(self):
        if self.debug:
            snake_game = SnakeGame(self.configs["game"]["board_size"])
            snake_player = SnakePlayer(
                self.configs["game"]["visualize"]["window_size"],
                self.configs["game"]["board_size"],
                self.configs["game"]["visualize"]["fps"],
            )
            snake_player.game_loop(snake_game, self)
        if self.fitness:
            print(f"Avg length: {self.fitness[0]}, Avg turns: {self.fitness[1]}")

    def calc_fitness(self):
        games = [SnakeGame(self.configs["game"]["board_size"]) for _ in range(self.simulation_times)]
        ended = [False for _ in range(self.simulation_times)]

        while True:
            batch = []
            idxs = []
            for i, game in enumerate(games):
                if not ended[i]:
                    batch.append(SnakeIndividualNN.encode_current_position(game))
                    idxs.append(i)
            if len(batch) == 0:
                break
            batch = torch.stack(batch).to(self.device)
            prob = self.chromosome(batch).softmax(1)
            for i, idx in enumerate(idxs):
                action = prob[i].argmax(0).item()
                if action == 0:
                    games[idx].velocity = (-1, 0)
                elif action == 1:
                    games[idx].velocity = (1, 0)
                elif action == 2:
                    games[idx].velocity = (0, -1)
                else:
                    games[idx].velocity = (0, 1)
                if games[idx].update() != SnakeGame.GAME_RUNNING:
                    ended[idx] = True
        all_scores = [len(game.snake) for game in games]
        all_turns = [game.turns for game in games]
        self.fitness = (np.mean(all_scores), np.mean(all_turns))

    def __lt__(self, other):
        if self.fitness[0] == other.fitness[0]:
            return self.fitness[1] > other.fitness[1]
        return self.fitness[0] < other.fitness[0]


class SnakeGANN(GeneticAlgorithmNN):
    INDIVIDUAL_CLASS = SnakeIndividualNN
    NN_CLASS = SnakeNN

    def __init__(self, configs: dict, pretrained_weights: str = ''):
        super().__init__(configs, pretrained_weights)

    def can_terminate(self, evolved, gen):
        return gen >= self.max_gen or \
            self.goat.fitness[0] >= self.configs["game"]["board_size"][0] * self.configs["game"]["board_size"][1]

