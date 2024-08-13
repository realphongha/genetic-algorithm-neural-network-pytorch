import numpy as np
import yaml
import torch

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

    def __init__(self, configs, network_class, network=None):
        self.simulation_times = configs["simulation_times"]
        super().__init__(configs, network_class, network)
        # how many times to run simulation

    @staticmethod
    def encode_current_position2(game):
        w, h = game.w, game.h
        food = game.food
        head = game.snake[-1]
        previous_head = game.snake[-2]
        tail = game.snake[0]
        snake_length = len(game.snake)
        inp = [
            1/w, 1/h,
            food[0]/w, food[1]/h,
            head[0]/w, head[1]/h,
            previous_head[0]/w, previous_head[1]/h,
            tail[0]/w, tail[1]/h,
            snake_length / w / h
        ]

        inp = torch.tensor(inp).float()
        return inp

    @staticmethod
    def encode_current_position1(game):
        w, h = game.w, game.h
        food = game.food[1]*w + game.food[0]
        head = game.snake[-1][1]*w + game.snake[-1][0]
        head2 = game.snake[-2][1]*w + game.snake[-2][0]
        head3 = game.snake[-3][1]*w + game.snake[-3][0]
        tail = game.snake[0][1]*w + game.snake[0][0]
        snake_len = len(game.snake)
        inp = [
            w, h, tail, head3, head2, head, food, snake_len
        ]
        # inp = [
        #     w, h, # board size
        #     game.food[1]*w + game.food[0],  # food
        #     game.snake[-1][1]*w + game.snake[-1][0],  # head
        #     game.snake[0][1]*w + game.snake[0][0],  # tail
        #     len(game.snake)  # snake length
        # ]
        inp = torch.tensor(inp).float()
        # normalize by board size
        return (inp-(w*h)/2) / w / h

    @staticmethod
    def encode_current_position(game):
        # encode the entire board, each row as an number
        w, h = game.w, game.h
        # empty cell - encoded as 0
        inp = torch.zeros(h)
        # snake segments - encoded as 1
        for x, y in game.snake[:-1]:
            inp[y] += 4 ** x
        # snake head - encoded as 2
        x, y = game.snake[-1]
        inp[y] += 2 * (4 ** x)
        # food - encoded as 3
        x, y = game.food
        inp[y] += 3 * (4 ** x)
        head = game.snake[-1]
        # normalize input
        inp /= 4 ** w
        return inp

    @torch.no_grad()
    def get_action(self, game):
        prob = self.chromosome(SnakeIndividualNN.encode_current_position(game).to(self.device))
        direction = prob.argmax(0).item()
        if direction == 0: # go ahead
            return game.velocity
        elif direction == 1:  # turn left
            return SnakeIndividualNN.TURN_LEFT[game.velocity]
        else:  # direction == 2:  # turn right
            return SnakeIndividualNN.TURN_RIGHT[game.velocity]

    def display(self):
        if self.debug:
            snake_game = SnakeGame(configs["game"]["board_size"])
            snake_player = SnakePlayer(
                configs["game"]["visualize"]["window_size"],
                configs["game"]["board_size"],
                configs["game"]["visualize"]["fps"],
            )
            snake_player.game_loop(snake_game, self)
        print(f"Avg length: {self.fitness[0]}, Avg turns: {self.fitness[1]}")

    def calc_fitness(self):
        all_scores = []
        all_turnes = []
        for _ in range(self.simulation_times):
            snake_game = SnakeGame(configs["game"]["board_size"])
            res = SnakeGame.GAME_RUNNING
            turns = -1
            turns_without_food = 0
            last_len = len(snake_game.snake)
            while res == SnakeGame.GAME_RUNNING:
                turns += 1
                snake_game.velocity = self.get_action(snake_game)
                res = snake_game.update()
                if len(snake_game.snake) == last_len:
                    turns_without_food += 1
                else:
                    turns_without_food = 0
                    last_len = len(snake_game.snake)
                if turns_without_food > snake_game.w + snake_game.h:
                    break
            all_scores.append(len(snake_game.snake))
            all_turnes.append(turns)
        return np.average(all_scores), np.average(all_turnes)

    def __lt__(self, other):
        if self.fitness[0] == other.fitness[0]:
            return self.fitness[1] > other.fitness[1]
        return self.fitness[0] < other.fitness[0]


class SnakeGANN(GeneticAlgorithmNN):
    def __init__(self, configs: dict):
        super().__init__(configs)

    def init_population(self):
        for _ in range(self.population_size):
            self.population.append(SnakeIndividualNN(self.configs, SnakeNN))

    def can_terminate(self, evolved, gen):
        return gen >= self.max_gen


if __name__ == "__main__":
    configs = yaml.load(open("snake/configs.yaml"), Loader=yaml.FullLoader)
    if configs["device"] == "cuda" and not torch.cuda.is_available():
        configs["device"] = "cpu"
    if configs["train"]:
        snake = SnakeGANN(configs)
        snake.run()
    if configs["test"]:
        configs["debug"] = True
        goat = SnakeIndividualNN(configs, SnakeNN)
        goat.load_weights(configs["save_path"])
        goat.display()

