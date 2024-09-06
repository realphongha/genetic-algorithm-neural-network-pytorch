# genetic-algorithm-neural-network-pytorch
Implementation for Genetic algorithm with neural network in Pytorch.

# Abstract classes
- `genetic_algorithm.py`: Abstract class for genetic algorithm.
- `genetic_algorithm_neural_network.py`: Abstract class for genetic algorithm 
with neural networks.

# Xor calculation
- A simple example for genetic algorithm neural network to 
calculate XOR operation.
- To train the model: 
```bash
python train.py --game xor --cfg xor_calculation/configs.yaml
```
- To test the model:
```bash
python test.py --game xor --cfg xor_calculation/configs.yaml --weights weights/xor/best.pth
```

# Dino bot
- A bot to play the Chrome Dino game with genetic algorithm neural network.
- If train correctly this bot can play the game perfectly.
- To play the Dino game by yourself:
```bash
python -m dino.dino
```
- To test our pre-trained model:
```bash
python test.py --game dino --weights weights/dino.pth --cfg dino/configs.yaml
```
- To train the bot:
```bash
python train.py --game dino --cfg dino/configs.yaml --no-debug
```

# Snake bot
- A simple bot to play the Snake game with genetic algorithm neural network.
- The input state of the network is still very simple, so at best the bot will
  work like A* search.
- To play the Snake game by yourself:
```bash
python -m snake.snake
```
- To test our pre-trained model:
```bash
python test.py --game snake --weights weights/snake.pth --cfg snake/configs.yaml
```
- To train the bot:
```bash
python train.py --game snake --cfg snake/configs.yaml --no-debug

