import pygame
import numpy as np
from pygame.locals import *
from envs.deep_sea_treasure.deep_sea_treasure import DeepSeaTreasure
import torch

# Constants for visualization
CELL_SIZE = [1.2 * 50, 0.8 * 50]
WINDOW_WIDTH = 10 * CELL_SIZE[0]
WINDOW_HEIGHT = 11 * CELL_SIZE[1]
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

class DeepSeaTreasureVisualization:
    def __init__(self, env, agent=None, device='cpu'):
        self.env = env
        self.agent = agent
        self.device = device
        self.root = './envs/deep_sea_treasure'
        self.state = None

        pygame.init()
        self.clock = pygame.time.Clock()
        self.display_surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.font = pygame.font.SysFont('Arial', 22)
        pygame.display.set_caption("Deep Sea Treasure")

        # get treasure image
        self.t_img = pygame.image.load(self.root + '/res/treasure.png')
        self.t_img = pygame.transform.scale(self.t_img, (0.95*CELL_SIZE[0], 0.95*CELL_SIZE[1]))
        self.t_img.convert()
        # get submarine image
        self.s_img = pygame.image.load(self.root + '/res/submarine.png')
        self.s_img = pygame.transform.scale(self.s_img, (0.9 * CELL_SIZE[0], 0.9 * CELL_SIZE[1]))
        self.s_img.convert()

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
                    self.display_surface.blit(self.t_img, treasure_rect)
                    self.display_surface.blit(self.font.render(str(value), True, BLACK), value_rect)

                pygame.draw.rect(self.display_surface, BLACK, rect, 1)

    def draw_agent(self): 
        row, col = self.state
        print(f'Current state is [{row} {col}]')
        # centering
        agent_rect = pygame.Rect(col * CELL_SIZE[0], row * CELL_SIZE[1], CELL_SIZE[0], CELL_SIZE[1])
        # display
        self.display_surface.blit(self.s_img, agent_rect)

    def update_display(self):
        self.display_surface.fill(WHITE)
        self.draw_map()
        self.draw_agent()
        pygame.display.update()
        pygame.time.wait(300) # pause for 0.3 second for each movement
        
    def play(self):
        print('------------------------------- NEW GAME -------------------------------')
        self.state = self.env.reset()
        self.update_display()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False

            # select action
            if self.agent == None:
                action = [np.random.randint(self.env.action_spec[2][0], self.env.action_spec[2][1])]
            else:
                action, _ = self.agent.select_action(torch.tensor(self.state, dtype=torch.float, device=self.device), 0)
            self.state, _, done = self.env.step(action[0])
            self.update_display()

            if done:
                print(f'Done! Get Treasure {_[0]}')
                print('------------------------------- END GAME -------------------------------')
                pygame.time.wait(1000)  # pause for 1.0 second before resetting
                print('------------------------------- NEW GAME -------------------------------')
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