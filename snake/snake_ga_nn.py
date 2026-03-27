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
        vx, vy = game.velocity
        
        # 8 relative directions: front, front-right, right, back-right, back, back-left, left, front-left
        directions = [
            (vx, vy),
            (vx - vy, vy + vx),
            (-vy, vx),
            (-vx - vy, -vy + vx),
            (-vx, -vy),
            (-vx + vy, -vy - vx),
            (vy, -vx),
            (vx + vy, vy - vx)
        ]
        
        tail = game.snake.popleft()
        vision = []
        for dx, dy in directions:
            wall_dist = 0.0
            food_found = 0.0
            body_dist = 0.0
            
            curr_x, curr_y = head[0], head[1]
            dist = 0.0
            
            found_food = False
            found_body = False
            
            while True:
                dist += 1.0
                curr_x += dx
                curr_y += dy
                
                if curr_x < 0 or curr_x >= game.w or curr_y < 0 or curr_y >= game.h:
                    wall_dist = 1.0 / dist
                    break
                
                if not found_food and (curr_x, curr_y) == game.food:
                    food_found = 1.0
                    found_food = True
                    
                if not found_body and (curr_x, curr_y) in game.snake:
                    body_dist = 1.0 / dist
                    found_body = True
            
            vision.extend([wall_dist, food_found, body_dist])
            
        game.snake.appendleft(tail)
        
        return torch.tensor(vision).float()

    @torch.no_grad()
    def get_action(self, game):
        prob = self.chromosome(SnakeIndividualNN.encode_current_position(game).to(self.device))
        action = prob.argmax(0).item()
        vx, vy = game.velocity
        if action == 0: # Turn Left
            return (vy, -vx)
        elif action == 2: # Turn Right
            return (-vy, vx)
        else: # Go Straight (1)
            return (vx, vy)

    def display(self):
        if self.debug:
            snake_game = SnakeGame(self.configs["game"]["board_size"])
            snake_player = SnakePlayer(
                self.configs["game"]["visualize"]["window_size"],
                self.configs["game"]["board_size"],
                self.configs["game"]["visualize"]["fps"],
            )
            snake_player.game_loop(snake_game, self)
        if self.fitness is not None:
            print(f"Fitness: {self.fitness}")

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
                vx, vy = games[idx].velocity
                if action == 0:
                    games[idx].velocity = (vy, -vx)
                elif action == 2:
                    games[idx].velocity = (-vy, vx)
                else:
                    games[idx].velocity = (vx, vy)
                if games[idx].update() != SnakeGame.GAME_RUNNING:
                    ended[idx] = True
        
        self.fitness = np.mean([
            (len(game.snake) - 3) * 10000 + game.turns for game in games
        ])

    def __lt__(self, other):
        return self.fitness < other.fitness


class SnakeGANN(GeneticAlgorithmNN):
    INDIVIDUAL_CLASS = SnakeIndividualNN
    NN_CLASS = SnakeNN

    def __init__(self, configs: dict, pretrained_weights: str = ''):
        super().__init__(configs, pretrained_weights)

    def can_terminate(self, evolved, gen):
        return gen >= self.max_gen or self.goat.fitness >= (self.configs["game"]["board_size"][0] * self.configs["game"]["board_size"][1] - 3) * 10000

