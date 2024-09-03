import argparse
import yaml

import torch

from dino import DinoGANN
from snake import SnakeGANN
from xor_calculation import XorCalculationGANN

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
                        help='Path to configs file')
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
    if args.game == 'dino':
        game = DinoGANN(configs, args.weights)
    elif args.game == 'snake':
        game = SnakeGANN(configs, args.weights)
    elif args.game == 'xor':
        game = XorCalculationGANN(configs)
    else:
        raise NotImplementedError(f"{args.game} not implemented!")
    game.run()

