import numpy as np
import random
from typing import Optional


class SnakePlayer:
    def __init__(self, window_size=(400, 400), board_size=(8, 8), fps=5):
        import pygame

        self.pygame = pygame
        self.pygame.init()
        self.window_w, self.window_h = window_size
        self.board_w, self.board_h = board_size
        assert (self.window_w % self.board_w == 0)
        assert (self.window_h % self.board_h == 0)
        self.block_w = self.window_w // self.board_w
        self.block_h = self.window_h // self.board_h
        self.window_size = window_size
        self.pygame.display.set_caption("Snake game. Press 'Q' to quit")
        self.screen = self.pygame.display.set_mode(window_size)
        self.background_color = (0, 0, 0)
        self.line_color = (211, 211, 211)
        self.snake_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        self.snake_color = random.choice(self.snake_colors)
        self.food_color = (255, 255, 255)
        self.font = self.pygame.font.Font(None, 25)
        self.font_color = (255, 255, 255)
        self.fps = fps
        self.clock = self.pygame.time.Clock()

    def change_snake_color(self):
        new_color = random.choice(self.snake_colors)
        while new_color == self.snake_color:
            new_color = random.choice(self.snake_colors)
        self.snake_color = new_color

    def draw_board(self):
        # fills background
        self.screen.fill(self.background_color)

    def draw_snake_and_food(self, snake, food):
        # draws snake
        for w, h in snake:
            self.pygame.draw.rect(self.screen, self.snake_color,
                self.pygame.Rect(w * self.block_w, h * self.block_h,
                                 self.block_w, self.block_h))
        # draws food
        w, h = food
        self.pygame.draw.rect(self.screen, self.food_color,
            self.pygame.Rect(w * self.block_w, h * self.block_h,
                             self.block_w, self.block_h))

    def game_over_screen(self, won=False):
        while True:
            # captures events
            for event in self.pygame.event.get():
                if event.type == self.pygame.QUIT:
                    # quits game
                    print("Quit")
                    return "quit"
                elif event.type == self.pygame.KEYDOWN:
                    if event.key == self.pygame.K_r:
                        return "restart"
                    elif event.key == self.pygame.K_q:
                        # quits game
                        print("Quit")
                        return "quit"

            # draws board
            self.draw_board()

            # draws text
            text = self.font.render(
                f"{'Won!' if won else 'Game over!'} Press 'R' to restart!",
                True, self.font_color)
            text_rect = text.get_rect(
                center=(self.window_w//2, self.window_h//2))
            self.screen.blit(text, text_rect)

            # updates screen
            self.pygame.display.update()
            self.clock.tick(self.fps)

    def game_loop(self, game, bot=None):
        while True:
            if random.random() < 0.05:
                self.change_snake_color()
            if bot is None: # human player
                for event in self.pygame.event.get():
                    if event.type == self.pygame.QUIT:
                        # quits game
                        quit(0)
                    elif event.type == self.pygame.KEYDOWN:
                        if event.key == self.pygame.K_LEFT:
                            if game.velocity[0] != 1:
                                game.velocity = (-1, 0)
                        elif event.key == self.pygame.K_RIGHT:
                            if game.velocity[0] != -1:
                                game.velocity = (1, 0)
                        elif event.key == self.pygame.K_UP:
                            if game.velocity[1] != 1:
                                game.velocity = (0, -1)
                        elif event.key == self.pygame.K_DOWN:
                            if game.velocity[1] != -1:
                                game.velocity = (0, 1)
                        elif event.key == self.pygame.K_q:
                            # quits game
                            break
                res = game.update()
                if res != SnakeGame.GAME_RUNNING:
                    break
            else:
                game.velocity = bot.get_action(game)
                res = game.update()
                if res != SnakeGame.GAME_RUNNING:
                    break
            # draws
            self.draw_board()
            self.draw_snake_and_food(game.snake, game.food)
            # updates screen
            self.pygame.display.update()
            self.clock.tick(self.fps)

        # game over
        if bot is None:
            print("Game over!")
            command = self.game_over_screen(res == SnakeGame.GAME_WON)
            if command == "restart":
                print("Restart")
                game.init_new_game()
                self.game_loop(game, bot)


class SnakeGame:
    GAME_WON = 0
    GAME_OVER = 1
    GAME_RUNNING = 2

    def __init__(self, board_size=(8, 8)):
        self.w, self.h = board_size
        assert self.w >= 8, "Board width should be larger than 8"
        assert self.h >= 8, "Board height should be larger than 8"
        self.board = np.zeros((self.h, self.w), dtype=np.ubyte)
        self.snake = None
        self.velocity = None
        self.food = None
        self.init_new_game()

    def init_new_game(self):
        self.snake = [(self.w//2-2, self.h//2),
                      (self.w//2-1, self.h//2),
                      (self.w//2, self.h//2)]
        self.velocity = (1, 0)
        self.spawn_food()

    def spawn_food(self):
        self.food = (np.random.randint(0, self.w),
                     np.random.randint(0, self.h))
        while self.food in self.snake:
            self.food = (np.random.randint(0, self.w),
                         np.random.randint(0, self.h))

    def update(self):
        head = self.snake[-1]
        new_head = (head[0] + self.velocity[0], head[1] + self.velocity[1])
        if new_head[0] < 0 or new_head[0] >= self.w or \
            new_head[1] < 0 or new_head[1] >= self.h:
            return SnakeGame.GAME_OVER
        elif new_head in self.snake:
            return SnakeGame.GAME_OVER
        elif new_head == self.food:
            self.snake.append(new_head)
            if len(self.snake) == self.w * self.h:
                return SnakeGame.GAME_WON
            self.spawn_food()
        else:
            self.snake = self.snake[1:]
            self.snake.append(new_head)
        return SnakeGame.GAME_RUNNING


if __name__ == "__main__":
    snake_game = SnakeGame()
    snake_player = SnakePlayer(fps=3)
    snake_player.game_loop(snake_game)

