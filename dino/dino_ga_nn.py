import yaml
import argparse

import torch
import numpy as np

from genetic_algorithm_neural_network import GeneticAlgorithmNN, IndividualNN
from dino.dino_nn import DinoNN
from dino.dino import DinoPlayer, Dino


class DinoIndividualNN(IndividualNN):
    def __init__(self, configs, network_class, network=None):
        self.simulation_times = configs["simulation_times"]
        super().__init__(configs, network_class, network)
        # how many times to run simulation

    @staticmethod
    def encode_current_position(game):
        inp = [
            (game.h-game.dino[1]) / game.h, # dino y
            game.dino_size[1] / game.h, # dino h
            game.velocity_y / game.h, # dino velocity by y
            (game.obstacle.x - game.dino[0]) / game.w, # obstacle x
            (game.h-game.obstacle.y) / game.h, # obstacle y
            game.obstacle.w / game.w, # obstacle w
            game.obstacle.h / game.h, # obstacle h
            -game.speed / game.w # speed
        ]
        return torch.tensor(inp).float()

    @torch.no_grad()
    def get_action(self, game):
        prob = self.chromosome(DinoIndividualNN.encode_current_position(game).to(self.device))
        cls_id = prob.argmax(0).item()
        if cls_id == 1:
            return "jump"
        if cls_id == 2:
            return "duck"
        else: # == 0
            return "no_action"

    def display(self):
        print(f"Avg score: {self.fitness[0]}")
        if self.debug:
            dino_game = Dino(self.configs["game"])
            dino_player = DinoPlayer(self.configs["game"])
            dino_player.game_loop(dino_game, self)
            return dino_game.score
        return self.fitness[0]

    def calc_fitness(self):
        all_scores = []
        all_jump_times = []
        dino_game = Dino(self.configs["game"])
        for _ in range(self.simulation_times):
            dino_game.init_new_game()
            res = Dino.GAME_RUNNING
            while res == Dino.GAME_RUNNING:
                action = self.get_action(dino_game)
                if action == "jump":
                    dino_game.stand()
                    dino_game.jump()
                elif action == "duck":
                    dino_game.duck()
                else:
                    dino_game.stand()
                # remember to set win_score or game will never end if the bot is too good
                res = dino_game.update()

            all_scores.append(dino_game.score)
            all_jump_times.append(dino_game.jump_times)
        return np.average(all_scores), np.average(all_jump_times)

    def __lt__(self, other):
        if self.fitness[0] == other.fitness[0]:
            return self.fitness[1] > other.fitness[1]
        return self.fitness[0] < other.fitness[0]


class DinoGANN(GeneticAlgorithmNN):
    INDIVIDUAL_CLASS = DinoIndividualNN
    NN_CLASS = DinoNN

    def __init__(self, configs: dict, pretrained_weights: str = ''):
        super().__init__(configs, pretrained_weights)

    def can_terminate(self, evolved, gen):
        return gen >= self.max_gen or self.goat.fitness[0] >= self.configs["game"]["win_score"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',
                        type=str,
                        default='dino/configs.yaml',
                        help='path to configs file')
    parser.add_argument('--weights',
                        type=str,
                        default='',
                        help='Path to pre-trained weights file')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--no-debug', dest='debug', action='store_false')
    parser.set_defaults(debug=True)
    args = parser.parse_args()
    configs = yaml.load(open(args.cfg), Loader=yaml.FullLoader)
    if configs["device"] == "cuda" and not torch.cuda.is_available():
        configs["device"] = "cpu"
    configs["debug"] = args.debug
    print("Training Dino bot...")
    dino = DinoGANN(configs, args.weights)
    dino.run()

