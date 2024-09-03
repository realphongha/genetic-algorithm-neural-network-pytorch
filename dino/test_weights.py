import os
import argparse

from .dino_ga_nn import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    configs["game"]["win_score"] = float("inf")
    configs["debug"] = True
    goat = DinoIndividualNN(configs, DinoNN, calc_fitness=False)
    goat.load_weights(args.weights)
    goat.display()

