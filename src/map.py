from copy import deepcopy as dcopy
import random
import numpy as np
from uuid import uuid4


class Map(object):
    def __init__(self, configs):
        # generate a unique random id for the map
        self.height_min = configs['height-min']
        self.width_min = configs['width-min']
        self.height_max = configs['height-max']
        self.width_max = configs['width-max']
        self.min_num_turns = configs['min-num-turns']
        self.max_num_turns = configs['max-num-turns']
        self.num_castles = configs['num-castles']
        self.num_ponds = configs['num-ponds']
        self.min_num_agents = configs['min-num-agents']
        self.max_num_agents = configs['max-num-agents']
        self.n_marks = 0
        self.n_turns = 0
        
    def update_agent_coords_in_order(self):
        self.agent_coords_in_order = [[], []]
        for i in range(self.height):
            for j in range(self.width):
                if self.agents[0][i, j] == 1:
                    self.agent_coords_in_order[0].append((i, j))
                if self.agents[1][i, j] == 1:
                    self.agent_coords_in_order[1].append((i, j))
    
    def make_random_map(self):
        self.height = random.randint(self.height_min, self.height_max)
        self.width = random.randint(self.width_min, self.width_max)
        self.agents = np.zeros((2, self.height, self.width), dtype=np.int8)
        self.walls = np.zeros((2, self.height, self.width), dtype=np.int8)
        self.castles = np.zeros((self.height, self.width), dtype=np.int8)
        self.territories = np.zeros((2, self.height, self.width), dtype=np.int8)
        self.ponds = np.zeros((self.height, self.width), dtype=np.int8)
        self.n_turns = random.randint(self.min_num_turns, self.max_num_turns)
        self.remaining_turns = self.n_turns
        self.agent_coords_in_order = [[], []]
        self.agent_current_idx = 0
        
        slots = {}
        for i in range(self.height):
            for j in range(self.width):
                if i != self.height - i - 1 or j != self.width - j - 1:
                    slots[(i, j)] = True
        # generate random symmetric castle coords in range of self
        for i in range(self.num_castles):
            # generate random from slots
            (x, y) = random.choice(list(slots.keys()))
            self.castles[x, y] = 1
            del slots[(x, y)]
        
        # generate random symmetric castle coords in range of self
        for i in range(self.num_ponds):
            # generate random from slots
            (x, y) = random.choice(list(slots.keys()))
            self.ponds[x, y] = 1
            del slots[(x, y)]
            
        #generate random symmetric agents positions in range of self
        self.num_agents = random.randint(self.min_num_agents, self.max_num_agents)
        for i in range(self.num_agents):
            (x, y) = random.choice(list(slots.keys()))
            self.agents[0, x, y] = 1
            self.agent_coords_in_order[0].append((x, y))
            del slots[(x, y)]
            (x, y) = random.choice(list(slots.keys()))
            self.agents[1, x, y] = 1
            self.agent_coords_in_order[1].append((x, y))
            del slots[(x, y)]
        
    
    def get_agent_position(self, player_id, agent_id):
        return self.agent_pos[player_id][agent_id]
    
    def load(self, castles, wall_pos, agent_pos):
        self.castles = castles
        self.wall_pos = wall_pos
        self.agent_pos = agent_pos
        self.n_agents = len(agent_pos)
        for x, y in castles:
            self.castle_board[x, y] = castles[(x, y)]
        for x, y in wall_pos:
            self.wall_board[x, y] = 1
        for x, y in agent_pos[0]:
            self.conquered_board[0][x, y] = 1
        for x, y in agent_pos[1]:
            self.conquered_board[1][x, y] = 1
            
    def show_map(self):
        print('-' * self.width * 4 + '\n')
        print(self.wall_board)
        print('\n' + '-' * self.width * 4 + '\n')
        _board = np.zeros((self.height, self.width), dtype=str)
        _board.fill(' ')
        for coord in self.castles:
            _board[coord[0], coord[1]] = 'T'
        for coord in self.wall_pos:
            _board[coord[0], coord[1]] = 'W'
        for coord in self.agent_pos[0]:
            _board[coord[0], coord[1]] = 'A'
        for coord in self.agent_pos[1]:
            _board[coord[0], coord[1]] = 'B'
                
        print(np.array2string(_board, separator=' '))
        print('\n' + '-' * self.width * 4)
    
    def in_bounds(self, x, y):
        return x >= 0 and x < self.height and y >= 0 and y < self.width
    
    def is_empty(self, x, y):
        return self.map[0][x][y] == 0 and self.map[1][x][y] == 0
    
    def set(self, id, x, y, value = 1):
        self.map[id][x][y] = value
        if value != 0:
            self.n_marks += 1
            
    def is_fully(self):
        return self.n_marks == self.height * self.width
    
    def to_opp(self):
        self.map = np.flip(self.map, 0)
        return self
    
    def reset(self):
        self.map = np.zeros((2, self.height, self.width))
                              
    
    def flatten(self):
        self.map.reshape(-1, )
        
    def copy(self):
        return dcopy(self)
    
    def root90(self, k = 0):
        self.map[0] = np.rot90(self.map[0], k)
        self.map[1] = np.rot90(self.map[1], k)
    
    def fliplr(self):
        self.map[0] = self.map[0][:, ::-1]
        self.map[1] = self.map[1][:, ::-1]
    
    def log(self, flip = False, icon = ('X', 'O')):
        if flip:
            icon = (icon[1], icon[0])
        for i in range(self.height):
            row = ['-'] * self.width
            for j in range(self.width):
                if self.map[0][i][j] == 1:
                    row[j] = icon[0]
                elif self.map[1][i][j] == 1:
                    row[j] = icon[1]
            print(row)
        print('-----------')
        