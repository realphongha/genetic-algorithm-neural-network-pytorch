import yaml
import torch

from genetic_algorithm_neural_network import GeneticAlgorithmNN, IndividualNN
from snake.snake_nn import SnakeNN
from snake.snake import SnakePlayer, SnakeGame


class SnakeIndividualNN(IndividualNN):
    def __init__(self, configs, network_class, network=None):
        super().__init__(configs, network_class, network)

    @staticmethod
    def encode_current_position(game):
        w, h = game.w, game.h
        inp = [
            w, h, # board size
            game.food[1]*w + game.food[0],
            game.snake[-2][1]*w + game.snake[-2][0],  # previous head
            game.snake[-1][1]*w + game.snake[-1][0],  # head
            game.snake[0][1]*w + game.snake[0][0],  # tail
            len(game.snake)
        ]
        inp = torch.tensor(inp).float()
        # normalize by board size
        return inp / w / h

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
        else: # direction == 3
            return (0, 1)

    def display(self):
        if self.debug:
            snake_game = SnakeGame(configs["game"]["board_size"])
            snake_player = SnakePlayer(
                configs["game"]["visualize"]["window_size"],
                configs["game"]["board_size"],
                configs["game"]["visualize"]["fps"],
            )
            snake_player.game_loop(snake_game, self)
            print("Score:", len(snake_game.snake))
        else:
            print("Score:", self.fitness)

    def calc_fitness(self):
        snake_game = SnakeGame(configs["game"]["board_size"])
        snake_game.velocity = self.get_action(snake_game)
        res = SnakeGame.GAME_RUNNING
        turns = -1
        turns_without_food = 0
        last_len = len(snake_game.snake)
        while res == SnakeGame.GAME_RUNNING:
            turns += 1
            res = snake_game.update()
            if len(snake_game.snake) == last_len:
                turns_without_food += 1
            else:
                turns_without_food = 0
                last_len = len(snake_game.snake)
            if turns_without_food > snake_game.w + snake_game.h:
                break
        return len(snake_game.snake)


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
    snake = SnakeGANN(configs)
    snake.run()

