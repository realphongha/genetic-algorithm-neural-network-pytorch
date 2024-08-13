import pygame
import random

# Initialize pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Dino Game")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Dino parameters
dino_width = 50
dino_height = 50
dino_x = 50
dino_y = SCREEN_HEIGHT - dino_height
dino_vel_y = 0
gravity = 1
jump_power = -15
is_jumping = False

# Obstacle parameters
obstacle_width = 20
obstacle_height = 50
obstacle_vel_x = -10
obstacle_x = SCREEN_WIDTH
obstacle_y = SCREEN_HEIGHT - obstacle_height

# Clock
clock = pygame.time.Clock()

# Main game loop
running = True
while running:
    screen.fill(WHITE)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Dino jump logic
    keys = pygame.key.get_pressed()
    if keys[pygame.K_SPACE] and not is_jumping:
        dino_vel_y = jump_power
        is_jumping = True

    dino_vel_y += gravity
    dino_y += dino_vel_y

    if dino_y >= SCREEN_HEIGHT - dino_height:
        dino_y = SCREEN_HEIGHT - dino_height
        is_jumping = False

    # Obstacle movement
    obstacle_x += obstacle_vel_x
    if obstacle_x < -obstacle_width:
        obstacle_x = SCREEN_WIDTH
        obstacle_height = random.randint(30, 70)
        obstacle_y = SCREEN_HEIGHT - obstacle_height

    # Collision detection
    if (dino_x + dino_width > obstacle_x and dino_x < obstacle_x + obstacle_width and
            dino_y + dino_height > obstacle_y):
        print("Game Over")
        running = False

    # Drawing Dino
    pygame.draw.rect(screen, BLACK, (dino_x, dino_y, dino_width, dino_height))

    # Drawing Obstacle
    pygame.draw.rect(screen, BLACK, (obstacle_x, obstacle_y, obstacle_width, obstacle_height))

    # Update the display
    pygame.display.flip()

    # Frame rate
    clock.tick(60)

# Quit pygame
pygame.quit()
