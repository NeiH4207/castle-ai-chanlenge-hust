# MODULES
import numpy as np
import pygame
import os 

RED = (255, 0, 0)
BG_COLOR = (245, 245, 245)
LINE_COLOR = (0, 0, 0)
CIRCLE_COLOR = (239, 231, 200)
CROSS_COLOR = (66, 66, 66)
    
class Screen():
    def __init__(self, render=True):
        if render:
            pygame.init()
            self.WIDTH = 500
            self.HEIGHT = 500
            self.LINE_WIDTH = 1
            self.SQUARE_SIZE = int(self.HEIGHT / 20)
            self.color_A = (255, 172,  88)
            self.color_B = (129, 188, 255)
            self.dir_path = os.path.dirname(os.path.realpath(__file__))
            self.load_image()
            pygame.display.set_caption( 'ProCon-2023' ) 
            self.board = None

    def init(self, state): 
        self.height = state.height
        self.width = state.width
        self.board = np.zeros((self.height, self.width), dtype=np.uint8)
        self.screen = pygame.display.set_mode(self.coord(self.height, self.width + 3))  
        self.screen.fill( BG_COLOR )
        self.draw_lines()
        self.load_state(state)
        if self.render:
            self.render()
        
    def render(self):
        pygame.display.update()
        
    def save(self, path):
        pygame.image.save(self.screen, path)

    def load_image(self):
        self.agent_A_img = pygame.transform.scale(
            pygame.image.load(self.dir_path + '/images/green_piece.png'), (self.SQUARE_SIZE, self.SQUARE_SIZE))
        self.cur_agent_img = pygame.transform.scale(
            pygame.image.load(self.dir_path + '/images/cur_piece.png'), (self.SQUARE_SIZE, self.SQUARE_SIZE))
        self.agent_B_img = pygame.transform.scale(
            pygame.image.load(self.dir_path + '/images/red_piece.png'), (self.SQUARE_SIZE, self.SQUARE_SIZE))
        self.wall_A_img =  pygame.transform.scale(
            pygame.image.load(self.dir_path + '/images/wall_green.png'), (self.SQUARE_SIZE, self.SQUARE_SIZE))
        self.wall_B_img =  pygame.transform.scale(
            pygame.image.load(self.dir_path + '/images/wall_red.png'), (self.SQUARE_SIZE, self.SQUARE_SIZE))
        self.background_img = pygame.transform.scale(
            pygame.image.load(self.dir_path + '/images/background.jpg'), (626, 986))
        self.table_img =  pygame.transform.scale(
            pygame.image.load(self.dir_path + '/images/board.png'), (512, 512))
        self.castle_img = pygame.transform.scale(
            pygame.image.load(self.dir_path + '/images/castle.png'), (self.SQUARE_SIZE, self.SQUARE_SIZE))
        self.pond_img = pygame.transform.scale(
            pygame.image.load(self.dir_path + '/images/pond.png'), (self.SQUARE_SIZE, self.SQUARE_SIZE))
        
    def load_state(self, state):
        # ID 0: Empty
        # ID 1: Wall A
        # ID 2: Wall B
        # ID 3: Agent A
        # ID 4: Agent B
        # ID 5: Territory A
        # ID 6: Territory B
        # ID 7: Castle
        # ID 8: Pond
        
        for i in range(self.height):
            for j in range(self.width):
                # draw wall
                if state.walls[0, i, j] == 1:
                    if self.board[i, j] != 1:
                        self.draw_wall(0, i, j)
                        self.board[i, j] = 1
                    continue
                elif state.walls[1, i, j] == 1:
                    if self.board[i, j] != 2:
                        self.draw_wall(1, i, j)
                        self.board[i, j] = 2
                    continue
                if state.castles[i, j] == 1:
                    if self.board[i, j] != 7:
                        self.draw_castle(i, j)
                        self.board[i, j] = 7
                    continue
                if state.ponds[i, j] == 1:
                    if self.board[i, j] != 8:
                        self.draw_pond(i, j)
                        self.board[i, j] = 8
                    continue
                if state.agents[0, i, j] == 1:
                    if self.board[i, j] != 3:
                        self.draw_agent(i, j, 0)
                        self.board[i, j] = 3
                    continue
                elif state.agents[1, i, j] == 1:
                    if self.board[i, j] != 4:
                        self.draw_agent(i, j, 1)
                        self.board[i, j] = 4
                    continue
                if state.territories[0, i, j] == 1:
                    if self.board[i, j] != 5:
                        self.draw_squares((i, j), 0)
                        self.board[i, j] = 5
                    continue
                elif state.territories[1, i, j] == 1:
                    if self.board[i, j] != 6:
                        self.draw_squares((i, j), 1)
                        self.board[i, j] = 6
                    continue
                        
                if self.board[i][j] != 0:
                    self.make_empty_square([i, j])
                    self.board[i, j] = 0
                
        self.show_score(state)
        
    def coord(self, x, y):
        return x * self.SQUARE_SIZE, y * self.SQUARE_SIZE
    
    def show_score(self, state):
        # self.screen.blit(self.table_img, self.coord(self.height - 1, -2))
        self.draw_rectangle((0, self.width), (self.height, self.width + 3), BG_COLOR)
        pygame.draw.line(self.screen, LINE_COLOR, self.coord(0, self.width), 
                              self.coord(self.height, self.width),
                              self.LINE_WIDTH )
        myFont = pygame.font.SysFont("Helvetica", 20)
        
        color = LINE_COLOR
        
        SA = myFont.render("Score: " + str(round(state.scores[0])), 0, color)
        SB = myFont.render("Score: " + str(round(state.scores[1])), 0, color)
        
        myFont = pygame.font.SysFont("Helvetica", 20)
        STurns = myFont.render("Steps left: " + str(state.remaining_turns), 0, color)
        
        text_1_coord = self.coord(1, self.width)
        text_2_coord = self.coord(1, self.width + 1)
        self.screen.blit(SA, (text_1_coord[0], text_1_coord[1] + 5))
        self.screen.blit(SB, (text_2_coord[0], text_2_coord[1] + 5))
        self.screen.blit(self.agent_A_img, self.coord(0, self.width))
        self.screen.blit(self.agent_B_img, self.coord(0, self.width + 1))
        # self.make_empty_square([0, self.width + 2])
        self.screen.blit(STurns, self.coord(0, self.width + 2))
    
    def show_value(self, value, x, y):
        myFont = pygame.font.SysFont("Times New Roman", 30)
        value = round(value)
        pos = 5
        if value >= 0 and value < 10:
            pos = 15
        elif value > 10 or value > -10:
            pos = 10
        value = myFont.render(str(value), 1, (0, 0, 0))
        self.screen.blit(value, (x * self.SQUARE_SIZE + pos, y * self.SQUARE_SIZE + 8))
        
    def draw_wall(self, player_id, x, y):
        if player_id == 0:
            self.screen.blit(self.wall_A_img, self.coord(x, y))
        else:
            self.screen.blit(self.wall_B_img, self.coord(x, y))
        
    def draw_castle(self, x, y):
        self.screen.blit(self.castle_img, self.coord(x, y))
        
    def draw_pond(self, x, y):
        self.screen.blit(self.pond_img, self.coord(x, y))
        
    def draw_rectangle(self, coord_1, coord_2, color=BG_COLOR):
        x1, y1 = self.coord(*coord_1)
        x2, y2 = self.coord(*coord_2)
        pygame.draw.rect(self.screen, color, (x1, y1, (x2 - x1), (y2 - y1)))
        
    def draw_agent(self, x, y, player_ID):
        player_img = self.agent_A_img if player_ID == 0 else self.agent_B_img
        self.screen.blit(player_img, self.coord(x, y))
        
    
    def draw_lines(self):
        for i in range(self.width + 1):
            pygame.draw.line(self.screen, LINE_COLOR, (0, i * self.SQUARE_SIZE), 
                              (self.height * self.SQUARE_SIZE, i * self.SQUARE_SIZE), 
                              self.LINE_WIDTH )
        for i in range(self.height):
            pygame.draw.line(self.screen, LINE_COLOR, (i * self.SQUARE_SIZE, 0),
                             (i * self.SQUARE_SIZE, self.width * self.SQUARE_SIZE), self.LINE_WIDTH )
    
    def _draw_squares(self, x1, y1, x2, y2, player_ID):
        color = self.color_A if player_ID == 0 else self.color_B
        pygame.draw.rect(self.screen, color, (x1, y1, x2, y2))
        
        
    def draw_squares(self, coord, player_ID):
        x, y = coord
        self._draw_squares(1 + x * self.SQUARE_SIZE, 1 + y * self.SQUARE_SIZE,
                           (self.SQUARE_SIZE - 1), (self.SQUARE_SIZE - 1), player_ID)
        
    def _redraw_squares(self, x1, y1, x2, y2, player_ID):
        color = self.color_A if player_ID == 0 else self.color_B
        pygame.draw.rect(self.screen, color, (x1, y1, x2, y2))
        
        
    def redraw_squares(self, x, y, player_ID):
        self._redraw_squares(1 + x * self.SQUARE_SIZE, 1 + y * self.SQUARE_SIZE,
                           (self.SQUARE_SIZE - 1), (self.SQUARE_SIZE - 1), player_ID)
        self.show_value(self.state.title_board[x][y], x, y)
           
    def _make_empty_squares(self, x1, y1, x2, y2):
        color = BG_COLOR
        pygame.draw.rect(self.screen, color, (x1, y1, x2, y2))
        
    def make_empty_square(self, coord):
        x, y = coord
        self._make_empty_squares(1 + x * self.SQUARE_SIZE, 1 + y * self.SQUARE_SIZE,
                           (self.SQUARE_SIZE - 1), (self.SQUARE_SIZE - 1))
        