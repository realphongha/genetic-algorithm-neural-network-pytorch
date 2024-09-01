import random
from collections import deque

import numpy as np


class DinoPlayer:
    def __init__(self, configs):
        import pygame
        self.configs = configs
        self.pygame = pygame
        self.pygame.init()
        self.pygame.display.set_caption("Dino game. Press 'Q' to quit")
        self.screen = self.pygame.display.set_mode(configs["screen_size"])
        self.background_color = (0, 0, 0)
        self.line_color = (211, 211, 211)
        self.dino_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        self.dino_color = random.choice(self.dino_colors)
        self.obs_color = (255, 255, 255)
        self.font = self.pygame.font.Font(None, 25)
        self.font_color = (255, 255, 255)
        self.fps = configs["fps"]
        self.clock = self.pygame.time.Clock()

    def draw_board(self):
        self.screen.fill(self.background_color)

    def draw_dino_and_obstacle(self, dino, obstacle):
        self.pygame.draw.rect(self.screen, self.dino_color,
            self.pygame.Rect(*dino, *self.configs["dino_size"]))
        self.pygame.draw.rect(self.screen, self.obs_color,
            self.pygame.Rect(obstacle.x, obstacle.y, obstacle.w, obstacle.h))

    def draw_score(self, score):
        text = self.font.render(f"Score: {score}", True, self.font_color)
        self.screen.blit(text, (50, 50))

    def game_loop(self, dino_game, bot=None):
        while True:
            if random.random() < 0.005:
                self.dino_color = random.choice(self.dino_colors)
            if bot:
                action = bot.get_action(dino_game)
                if action == "jump":
                    dino_game.stand()
                    dino_game.jump()
                elif action == "duck":
                    dino_game.duck()
                else:
                    dino_game.stand()
            else:
                for event in self.pygame.event.get():
                    if event.type == self.pygame.QUIT:
                        quit(0)
                    elif event.type == self.pygame.KEYDOWN:
                        if event.key in (self.pygame.K_SPACE, self.pygame.K_UP):
                            dino_game.stand()
                            dino_game.jump()
                        elif event.key == self.pygame.K_DOWN:
                            dino_game.duck()
                        elif event.key == self.pygame.K_q:
                            quit(0)
                    elif event.type == self.pygame.KEYUP:
                        if event.key == self.pygame.K_DOWN:
                            dino_game.stand()
            res = dino_game.update()
            if res != Dino.GAME_RUNNING:
                break
            self.draw_board()
            self.draw_dino_and_obstacle(dino_game.dino, dino_game.obstacle)
            self.draw_score(dino_game.score)
            self.pygame.display.update()
            self.clock.tick(self.fps)


class Obstacle:
    def __init__(self, configs):
        self.x, self.y = configs["screen_size"]
        self.type = random.choice(configs["obstacle"]["types"])
        cfgs = configs["obstacle"][self.type]
        self.w = random.randint(*cfgs["w"])
        self.h = random.randint(*cfgs["h"])
        self.y -= random.randint(*cfgs["y"])
        self.y -= self.h


class Dino:
    GAME_RUNNING = 0
    GAME_OVER = 1
    WON = 2

    def __init__(self, configs):
        self.configs = configs
        self.w, self.h = configs["screen_size"]
        self.jump_power = configs["jump_power"]
        self.acceleration = None
        self.gravity = configs["gravity"]
        self.min_dist = self.w // 4  # min distance between cacti and birds
        self.dino_size = None
        self.dino = None  # dino position
        self.velocity_y = None  # dino vertical velocity
        self.is_jumping = False
        self.action_count = None
        self.speed = None  # obstacle speed, image the dino stands still and obstacles keeps running to it
        self.obstacle = None
        self.score = None
        self.init_new_game()

    def init_new_game(self):
        self.dino_size = self.configs["dino_size"].copy()
        self.dino = np.array([50, self.h])
        self.velocity_y = 0
        self.action_count = 0
        self.is_jumping = False
        self.speed = self.configs["init_speed"]
        self.obstacle = Obstacle(self.configs)
        self.score = 0

    def update_dino(self):
        self.velocity_y += self.gravity
        self.dino[1] += self.velocity_y
        if self.dino[1] > self.h - self.dino_size[1]:
            self.dino[1] = self.h - self.dino_size[1]
            self.is_jumping = False
            self.velocity_y = 0

    def update_obstacle(self):
        if self.obstacle is None or self.obstacle.x < -self.obstacle.w:
            self.obstacle = Obstacle(self.configs)
        else:
            self.obstacle.x += self.speed

    def update_speed(self):
        self.score -= self.speed
        self.speed = self.configs["init_speed"] - \
            (self.score // self.configs["change_speed_each"]) * self.configs["accel"]
        if self.speed < self.configs["max_speed"]:
            self.speed = self.configs["max_speed"]

    def is_dead(self):
        # collision detection
        return not (
            self.dino[0] > self.obstacle.x + self.obstacle.w or
            self.dino[0] + self.dino_size[0] < self.obstacle.x or
            self.dino[1] > self.obstacle.y + self.obstacle.h or
            self.dino[1] + self.dino_size[1] < self.obstacle.y
        )

    def jump(self):
        if not self.is_jumping:
            self.velocity_y = self.jump_power
            self.is_jumping = True
            self.action_count += 1

    def duck(self):
        self.action_count += 1
        self.dino_size[1] = self.configs["duck_height"]

    def stand(self):
        self.dino_size[1] = self.configs["dino_size"][1]

    def update(self):
        self.update_dino()
        self.update_obstacle()
        self.update_speed()
        if self.is_dead():
            return Dino.GAME_OVER
        if self.score >= self.configs["win_score"]:
            return Dino.WON
        return Dino.GAME_RUNNING


if __name__ == "__main__":
    import yaml
    configs = yaml.load(open("dino/configs.yaml"), Loader=yaml.FullLoader)
    dino_game = Dino(configs["game"])
    dino_player = DinoPlayer(configs["game"])
    dino_player.game_loop(dino_game)

