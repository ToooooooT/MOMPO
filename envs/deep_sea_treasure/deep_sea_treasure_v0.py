import pygame
import numpy as np
from pygame.locals import *
from deep_sea_treasure import DeepSeaTreasure

# Constants for visualization
CELL_SIZE = [1.2 * 50, 0.8 * 50]
WINDOW_WIDTH = 10 * CELL_SIZE[0]
WINDOW_HEIGHT = 11 * CELL_SIZE[1]
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

class DeepSeaTreasureVisualization:
    def __init__(self, env):
        self.env = env
        self.state = None
        
        pygame.init()
        self.clock = pygame.time.Clock()
        self.display_surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.font = pygame.font.SysFont('Arial', 22)
        pygame.display.set_caption("Deep Sea Treasure")

    def draw_map(self):
        for row in range(self.env.sea_map.shape[0]):
            for col in range(self.env.sea_map.shape[1]):
                value = self.env.sea_map[row, col]
                rect = pygame.Rect(col * CELL_SIZE[0], row * CELL_SIZE[1], CELL_SIZE[0], CELL_SIZE[1])

                if value == 0:
                    pygame.draw.rect(self.display_surface, WHITE, rect)
                elif value == -10:
                    pygame.draw.rect(self.display_surface, BLACK, rect)
                else:
                    # get treasure image
                    img = pygame.image.load('./res/treasure.png')
                    img = pygame.transform.scale(img, (0.95*CELL_SIZE[0], 0.95*CELL_SIZE[1]))
                    img.convert()
                    # centering
                    treasure_rect = pygame.Rect(col * CELL_SIZE[0], row * CELL_SIZE[1], CELL_SIZE[0], CELL_SIZE[1])
                    treasure_rect.center = (col + 0.5) * CELL_SIZE[0], (row + 0.5) * CELL_SIZE[1]
                    if value >= 11.5:
                        value_rect = pygame.Rect(col * CELL_SIZE[0], row * CELL_SIZE[1], 
                                                    0.5*(CELL_SIZE[0]+30), 0.5*(CELL_SIZE[1]))
                    else:    
                        value_rect = pygame.Rect(col * CELL_SIZE[0], row * CELL_SIZE[1], 
                                                    0.5*CELL_SIZE[0], 0.5*CELL_SIZE[1])
                    value_rect.center = (col + 0.5) * CELL_SIZE[0], (row + 0.5) * CELL_SIZE[1]
                    # display
                    self.display_surface.blit(img, treasure_rect)
                    self.display_surface.blit(self.font.render(str(value), True, BLACK), value_rect)

                pygame.draw.rect(self.display_surface, BLACK, rect, 1)

    def draw_agent(self): 
        row, col = self.state
        # get submarine image
        img = pygame.image.load('./res/submarine.png')
        img = pygame.transform.scale(img, (0.9 * CELL_SIZE[0], 0.9 * CELL_SIZE[1]))
        img.convert()
        # centering
        agent_rect = pygame.Rect(col * CELL_SIZE[0], row * CELL_SIZE[1], CELL_SIZE[0], CELL_SIZE[1])
        agent_rect.center = (col//2 + 0.5) * CELL_SIZE[0], (row//2 + 0.5) * CELL_SIZE[1]
        # display
        self.display_surface.blit(img, agent_rect)

    def update_display(self):
        self.display_surface.fill(WHITE)
        self.draw_map()
        self.draw_agent()
        pygame.display.update()
        pygame.time.wait(300) # pause for 0.3 second for each movement
        
    def play(self):
        self.state = self.env.reset()
        self.update_display()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False

            # select action
            action = np.random.randint(self.env.action_spec[2][0], self.env.action_spec[2][1])
            self.state, _, done = self.env.step(action)
            self.update_display()

            if done:
                pygame.time.wait(300)  # pause for 0.3 second before resetting
                self.state = self.env.reset()
                self.update_display()
            
            self.clock.tick(FPS)

        pygame.quit()

def main():
    env = DeepSeaTreasure()
    game = DeepSeaTreasureVisualization(env)
    game.play()

if __name__ == '__main__':
    main()