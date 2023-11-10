
import numpy as np
from src.map import Map
from copy import deepcopy as dcopy

class State(Map):
    def __init__(self, configs, action_space):
        super().__init__(configs)
        self.action_space = action_space
        self.num_players = 2
        self.current_player = 0
        self.num_agents = None
        self.wall_scores = [0 for _ in range(self.num_players)]
        self.castle_scores = [0 for _ in range(self.num_players)]
        self.open_territory_scores = [0 for _ in range(self.num_players)]
        self.closed_territory_scores = [0 for _ in range(self.num_players)]
        self.territory_scores = [0 for _ in range(self.num_players)]
        self.alpha = 1 # effect of wall
        self.beta = 20 # effect of castle
        self.gamma = 5 # effect of territory
        self.obs_range = configs['obs_range']
        
        self.action_map = {
            ('Move', 'U'): 0,
            ('Move', 'D'): 1,
            ('Move', 'L'): 2,
            ('Move', 'R'): 3,
            ('Move', 'UL'): 4,
            ('Move', 'UR'): 5,
            ('Move', 'DL'): 6,
            ('Move', 'DR'): 7,
            ('Change', 'U'): 8,
            ('Change', 'D'): 9,
            ('Change', 'L'): 10,
            ('Change', 'R'): 11,
            ('Stay', 'Stay'): 12
        }
        
        self.direction_map = {
            'U': (-1, 0),
            'D': (1, 0),
            'L': (0, -1),
            'R': (0, 1),
            'UL': (-1, -1),
            'UR': (-1, 1),
            'DL': (1, -1),
            'DR': (1, 1)
        }
        
        self.n_actions = len(self.action_map.values())
        
    def hash_arr(self, arr: np.ndarray):
        s = ''.join([str(x) for x in arr.flatten()])
        return s
        
    
    def string_representation(self):
        """
        Returns a hash code for string representation of the state
        """
        players = [self.current_player, self.current_player ^ 1]
        s = self.hash_arr(self.agents[players]) + self.hash_arr(self.walls[players]) + \
            self.hash_arr(self.castles) + self.hash_arr(self.territories[players])
        return hash(s)    
    
    def current_position(self):
        return self.agent_coords_in_order[self.current_player][self.agent_current_idx]
        
    def copy(self):
        return dcopy(self)
        
    def get_curr_player(self):
        return self.current_player
    
    def get_curr_agent(self):
        curr_player = self.current_player
        return self.agent_coords_in_order[curr_player][self.agent_current_idx]
        
    @property
    def scores(self):
        score_A = self.alpha * self.wall_scores[0] + self.beta * self.castle_scores[0] + \
            self.gamma * (self.open_territory_scores[0] + self.closed_territory_scores[0])
        score_B = self.alpha * self.wall_scores[1] + self.beta * self.castle_scores[1] + \
            self.gamma * (self.open_territory_scores[1] + self.closed_territory_scores[1])
        return np.array([score_A, score_B])
    
    
    def set_players(self, players):
        self.players = players
    
    def get_agent_position(self):
        return self.agent_pos[self.current_player]
    
    def to_opponent(self):
        state = dcopy(self)
        state.current_player ^= 1
        return state

    def transition_matrix(self, matrix, vector):
        result = []
        dx, dy = vector
        rows, cols = len(matrix), len(matrix[0])

        for i in range(rows):
            row = []
            for j in range(cols):
                new_i, new_j = i - dx, j - dy
                if 0 <= new_i < self.height and 0 <= new_j < self.width:
                    row.append(matrix[new_i][new_j])
                else:
                    row.append(-1)
            if row:
                result.append(row)

        return np.array(result)

    def terminal(self):
        return self.remaining_turns == 0
    
    def get_state(self, partial=True):
        """
        partial = True (default) if you want to get the partial state,
        the environment will return the a matrix of size (self.obs_range x 2 + 1) x (self.obs_range x 2 + 1) 
        cropped from the full state with the center cell being the current agent.
        The partial state is 
        [
            [first_agent_board_matrix],
            [first_wall_board_matrix],
            [first_territory_board_matrix],
            [second_agent_board_matrix],
            [second_wall_board_matrix],
            [second_territory_board_matrix],
            [castle_board_matrix],
            [pond_board_matrix],
        ]
        see function get_state() in src/state.py
        Using env.get_state(partial=False) if you want to get the full state,
        the full state is a matrix of size height x width (observation_shape)
        """
        # Standardized variable names to improve readability
        players = [self.current_player, self.current_player ^ 1]
        agent_board = self.agents[players]
        castle_board = self.castles
        pond_board = self.ponds
        wall_board = self.walls[players]
        territory_board = self.territories[players]
        agent_current_board = np.zeros(agent_board[0].shape)
        obs = np.stack(
            (
                agent_board[0], 
                wall_board[0], 
                territory_board[0], 
                agent_board[1],
                wall_board[1],
                territory_board[1],
                castle_board,
                pond_board
            ),
            axis=0
        )
        
        
        
        # Standardized variable names to improve readability and changed key name
        current_agent_idx = self.agent_current_idx
        current_agent_coord = self.agent_coords_in_order[self.current_player][current_agent_idx]
        
        if partial:
            # crop obs to obs_range
            transision_vector = (self.obs_range - 1 - current_agent_coord[0], 
                                self.obs_range - 1 - current_agent_coord[1])
            for i, matrix in enumerate(obs):
                obs[i] = self.transition_matrix(matrix, transision_vector)
                
            obs = obs[:, :self.obs_range*2-1, :self.obs_range*2-1]
            masked_obs = [(obs[0] != -1) * 1]
            obs = np.concatenate([obs, masked_obs], axis=0)
            agent_current_board[current_agent_coord[0], current_agent_coord[1]] = 1
        
        valid_actions = np.zeros(len(self.action_map.values()), dtype=bool)
        for action in list(self.action_map.values()):
            if self.is_valid_action(action):
                valid_actions[action] = True
        if sum(valid_actions) == 0:
            for action in list(self.action_map.values()):
                if self.is_valid_action(action, drop_self=True):
                    valid_actions[action] = True
            
        return {
            'player-id': self.current_player,
            'observation': obs, 
            'current-agent-id': current_agent_idx,
            'curr_agent_xy': dcopy(current_agent_coord),
            'valid_actions': valid_actions,
            'remaning_turns': self.remaining_turns,
            'hash_str': self.string_representation(),
            }

    def get_scores(self, player):
        """
        Recalculates the score of the current player based on current state
        """
        dx = [0, 0, 1, -1]
        dy = [1, -1, 0, 0]
        
        height = self.height
        width = self.width
        avail = np.zeros((height, width), dtype=int)
        st = []
        opponent = 1 - player
        
        def validate(x, y):
            # if the current land is out of table bounds
            if x >= height or x < 0 or y >= width or y < 0:
                return False
            # if the current land is visited
            if avail[x][y]:
                return False
            # if the current land is a wall
            if self.walls[player][x][y]:
                return False
            return True
        
        for i in range(height):
            if validate(i, 0):
                st.append((i, 0))
                avail[i][0] = 1
            if validate(i, width - 1):
                st.append((i, width - 1))
                avail[i][width - 1] = 1
        
        for j in range(width):
            if validate(0, j):
                st.append((0, j))
                avail[0][j] = 1
            if validate(height - 1, j):
                st.append((height - 1, j))
                avail[height - 1][j] = 1
            
        while len(st) > 0:
            x, y = st.pop()
            for i in range(4):
                nx = x + dx[i]
                ny = y + dy[i]
                if validate(nx, ny):
                    avail[nx][ny] = 1
                    st.append((nx, ny))
        
        wall_score = self.walls[player].sum()
        
        closed_territory_score = height * width - avail.sum() - wall_score

        castle_score = 0
        
        for i in range(height):
            for j in range(width):
                if self.walls[opponent][i][j] == 1 and self.territories[player][i][j] == 1:
                    self.territories[player][i][j] = 0
                if avail[i][j] == 0 and self.walls[player][i][j] == 0:
                    self.territories[player][i][j] = 1
                    if self.castles[i][j] == 1:
                        castle_score += 1
        
        all_territory_score = self.territories[player].sum()
        
        open_territory_score = all_territory_score - closed_territory_score
        

        return wall_score, closed_territory_score, open_territory_score, castle_score
    
    def update_score(self):
        """
        Updates the score of the current player based on current state
        """
        for player in range(self.num_players):
            wall_score, closed_territory_score, open_territory_score, castle_score = self.get_scores(player)
            self.wall_scores[player] = wall_score
            self.closed_territory_scores[player] = closed_territory_score
            self.open_territory_scores[player] = open_territory_score
            self.territory_scores[player] = open_territory_score + closed_territory_score
            self.castle_scores[player] = castle_score
            
        for player in range(self.num_players):
            self.players[player].scores = self.scores[player]
    
    def get_type_action(self, action):
        """
        Returns the type of the given action and the corresponding action list item.

        :param action: An integer representing the index of the action in the flattened action list.
        :return: A tuple of strings. The first string is the type of the action ('Move', 'Build', 'Destroy', or 'Stay').
                The second string is the corresponding item from the action list.
        """
        move_len = len(self.action_space['Move'])
        change_len = len(self.action_space['Change'])

        if action < move_len:
            return ('Move', self.action_space['Move'][action])
        elif action < move_len + change_len:
            return ('Change', self.action_space['Change'][action - move_len])
        else:
            return ('Stay',)
    
    def is_valid_action(self, action, drop_self=False):
        current_player = self.current_player
        agent_coords_in_order = self.agent_coords_in_order
        agent_current_idx = self.agent_current_idx
        current_position = agent_coords_in_order[current_player][agent_current_idx]
        
        valid = True
        action_type = self.get_type_action(action)
        if action_type[0] == 'Move':
            direction = action_type[1]
            next_position = (self.direction_map[direction][0] + current_position[0],
                        self.direction_map[direction][1] + current_position[1])
            if not self.in_bounds(next_position[0], next_position[1]):
                valid = False
                
            elif next_position in self.agent_coords_in_order[0] or \
                        next_position in self.agent_coords_in_order[1]:
                valid = False
                
            elif self.agents[current_player][next_position[0]][next_position[1]] == 1:
                ''' in turn (N agent actions at the same time), only one agent can move at an area, 
                    so the other agent is moved into the same area befores
                    agents save next coordinates but agent_coords_in_order is not updated to check this '''
                valid = False
                
            elif self.ponds[next_position[0]][next_position[1]] == 1:
                valid = False
                
            elif self.walls[0][next_position[0]][next_position[1]] == 1 \
                    or self.walls[1][next_position[0]][next_position[1]] == 1:
                valid = False
                
            elif self.castles[next_position[0]][next_position[1]] == 1:
                valid = False
    
        elif action_type[0] == 'Change':
            direction = action_type[1]
            next_position = (self.direction_map[direction][0] + current_position[0],
                        self.direction_map[direction][1] + current_position[1])
            if not self.in_bounds(next_position[0], next_position[1]):
                valid = False
                
            elif self.castles[next_position[0]][next_position[1]] == 1:
                valid = False
                
            elif self.ponds[next_position[0]][next_position[1]] == 1:
                valid = False
                
            elif next_position in self.agent_coords_in_order[0] or \
                        next_position in self.agent_coords_in_order[1]:
                valid = False
            elif not drop_self and self.walls[current_player][next_position[0]][next_position[1]] == 1:
                valid = False
        else:
            valid = False
                
        return valid
    
    
    def is_terminal(self):
        """
        Checks if the game has ended by evaluating if there are any remaining turns left.

        :return: Boolean value indicating whether or not the game has ended.
        """
        return self.remaining_turns == 0
    
    
    def next(self, action):
        action_type = self.get_type_action(action)
        current_player = self.current_player
        agent_current_idx = self.agent_current_idx
        agent_coords_in_order = self.agent_coords_in_order
        current_position = agent_coords_in_order[current_player][agent_current_idx]
        
        is_valid = self.is_valid_action(action, drop_self=True)
        
        if is_valid:
            if action_type[0] == 'Move':
                direction = action_type[1]
                next_position = (self.direction_map[direction][0] + current_position[0],
                            self.direction_map[direction][1] + current_position[1])
                
                self.agents[current_player][next_position[0]][next_position[1]] = 1
                self.agents[current_player][current_position[0]][current_position[1]] = 0
                
            elif action_type[0] == 'Change':
                direction = action_type[1]
                wall_coord = (self.direction_map[direction][0] + current_position[0],
                            self.direction_map[direction][1] + current_position[1])
                if self.walls[0][wall_coord[0]][wall_coord[1]] == 0 \
                            and self.walls[1][wall_coord[0]][wall_coord[1]] == 0:
                    self.walls[current_player][wall_coord[0]][wall_coord[1]] = 1
                
                else:
                    self.walls[0][wall_coord[0]][wall_coord[1]] = 0
                    self.walls[1][wall_coord[0]][wall_coord[1]] = 0
            else:
                pass
            
            self.update_score()
            
        self.agent_current_idx = (agent_current_idx + 1) % self.num_agents
        if self.agent_current_idx == 0:
            self.current_player = (self.current_player + 1) % self.num_players
            self.update_agent_coords_in_order()
            if self.current_player == 0:
                self.remaining_turns -= 1
                
        return is_valid
