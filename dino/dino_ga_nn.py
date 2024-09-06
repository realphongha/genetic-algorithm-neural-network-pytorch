import torch
import numpy as np

from genetic_algorithm_neural_network import GeneticAlgorithmNN, IndividualNN
from dino.dino_nn import DinoNN
from dino.dino import DinoPlayer, Dino


class DinoIndividualNN(IndividualNN):
    def __init__(self, configs, network_class, network=None, calc_fitness=True):
        # how many times to run simulation
        self.simulation_times = configs["simulation_times"]
        super().__init__(configs, network_class, network, calc_fitness)

    @staticmethod
    def encode_current_position(game):
        # inp = [
        #     (game.h-game.dino[1]) / game.h, # dino y
        #     game.dino_size[1] / game.h, # dino h
        #     -game.velocity_y / game.h, # dino velocity by y
        #     (game.obstacle.x - game.dino[0]) / game.w, # obstacle x
        #     (game.h-game.obstacle.y) / game.h, # obstacle y
        #     game.obstacle.w / game.w, # obstacle w
        #     game.obstacle.h / game.h, # obstacle h
        #     -game.speed / game.w # speed
        # ]
        inp = [
            (game.h-game.dino[1]), # dino y
            game.dino_size[1], # dino h
            -game.velocity_y, # dino velocity by y
            (game.obstacle.x - game.dino[0]), # obstacle x
            (game.h-game.obstacle.y), # obstacle y
            game.obstacle.w, # obstacle w
            game.obstacle.h, # obstacle h
            -game.speed # speed
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
        if self.fitness:
            print(f"Avg score: {self.fitness[0]}")
        if self.debug:
            dino_game = Dino(self.configs["game"])
            dino_player = DinoPlayer(self.configs["game"])
            dino_player.game_loop(dino_game, self)
            return dino_game.score

    @torch.no_grad()
    def calc_fitness(self):
        dino_games = [Dino(self.configs["game"]) for _ in range(self.simulation_times)]
        ended = [False for _ in range(self.simulation_times)]

        # run simulation using batch infer
        while True:
            batch = []
            idxs = []
            for i, dino_game in enumerate(dino_games):
                if not ended[i]:
                    batch.append(DinoIndividualNN.encode_current_position(dino_game))
                    idxs.append(i)
            if len(batch) == 0:
                break
            batch = torch.stack(batch).to(self.device)
            prob = self.chromosome(batch).softmax(1)
            for i, idx in enumerate(idxs):
                action = prob[i].argmax(0).item()
                if action == 1:
                    dino_games[idx].stand()
                    dino_games[idx].jump()
                elif action == 2:
                    dino_games[idx].duck()
                else: # == 0
                    dino_games[idx].stand()
                if dino_games[idx].update() != Dino.GAME_RUNNING:
                    ended[idx] = True

        all_scores = [game.score for game in dino_games]
        all_action_count = [game.action_count for game in dino_games]
        self.fitness = (np.average(all_scores), np.average(all_action_count))

    def calc_fitness_single_thread(self):
        all_scores = []
        all_action_count = []
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
            all_action_count.append(dino_game.action_count)
        self.fitness = (np.average(all_scores), np.average(all_action_count))

    def __lt__(self, other):
        if self.fitness[0] == other.fitness[0]:
            # if same score, prioritize less action count
            return self.fitness[1] > other.fitness[1]
        return self.fitness[0] < other.fitness[0]


class DinoGANN(GeneticAlgorithmNN):
    INDIVIDUAL_CLASS = DinoIndividualNN
    NN_CLASS = DinoNN

    def __init__(self, configs: dict, pretrained_weights: str = ''):
        super().__init__(configs, pretrained_weights)

    def can_terminate(self, evolved, gen):
        return gen >= self.max_gen or self.goat.fitness[0] >= self.configs["game"]["win_score"]


