import argparse
import yaml

import torch

from dino import DinoIndividualNN, DinoNN
from snake import SnakeIndividualNN, SnakeNN
from xor_calculation import XorIndividualNN, XorNN


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--game',
                        type=str,
                        default='dino',
                        choices=['snake', 'dino', 'xor'],
                        help='Game name')
    parser.add_argument('--cfg',
                        type=str,
                        default='dino/configs.yaml',
                        help='path to configs file')
    parser.add_argument('--weights',
                        type=str,
                        default='weights/dino/best.pth',
                        help='Path to weights file')
    args = parser.parse_args()
    configs = yaml.load(open(args.cfg), Loader=yaml.FullLoader)
    if configs["device"] == "cuda" and not torch.cuda.is_available():
        configs["device"] = "cpu"
    configs["debug"] = True
    if args.game == 'dino':
        configs["game"]["win_score"] = float("inf")
        ind = DinoIndividualNN(configs, DinoNN, calc_fitness=False)
    elif args.game == 'snake':
        ind = SnakeIndividualNN(configs, SnakeNN, calc_fitness=False)
    elif args.game == 'xor':
        ind = XorIndividualNN(configs, XorNN, calc_fitness=False)
    else:
        raise NotImplementedError(f"{args.game} not implemented!")
    ind.load_weights(args.weights)
    ind.display()

