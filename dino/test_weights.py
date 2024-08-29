from .dino_ga_nn import *


configs = yaml.load(open("dino/configs.yaml"), Loader=yaml.FullLoader)
if configs["device"] == "cuda" and not torch.cuda.is_available():
    configs["device"] = "cpu"
configs["game"]["win_score"] = float("inf")
configs["debug"] = True
goat = DinoIndividualNN(configs, DinoNN)
goat.load_weights(configs["save_path"])
score = goat.display()
print("Final score:", score)

