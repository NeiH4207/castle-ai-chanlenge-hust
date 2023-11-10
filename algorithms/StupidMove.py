import random
import numpy as np

class StupidMove():
    def __init__(self, n_actions: int = 4, num_agents: int = 2) -> None:
        self.n_actions = n_actions
        self.num_agents = num_agents
        
        self.action_space = {
            'Move': ['U', 'D', 'L', 'R', 'UL', 'UR', 'DL', 'DR'],
            'Change': ['U', 'D', 'L', 'R'],
            'Stay': 1
        }
        
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
        
        
    def get_action(self, state, epsilon=0.0):
        current_player_id = state['player-id']
        current_agent_id = state['current-agent-id']
        agent_board = state['observation'][[0, 3]]
        wall_board = state['observation'][[1, 4]]
        territory_board = state['observation'][[2, 5]]
        castle_board = state['observation'][6]
        pond_board = state['observation'][7]
        masked = state['observation'][8] # 0 if out of land, 1 if inside
        valid_actions = state['valid_actions']
        obs_size = masked.shape
        scores = [-1] * self.n_actions
        curr_x, curr_y = obs_size[0] // 2, obs_size[1] // 2
        for i in range(len(self.action_space['Move'])):
            if valid_actions[self.action_map[('Move', self.action_space['Move'][i])]] == 1:
                score = 0
                x, y = self.direction_map[self.action_space['Move'][i]]
                new_x, new_y = x + curr_x, y + curr_y
                if territory_board[0][new_x, new_y] == 1:
                    score = -0.1
                elif wall_board[0][new_x, new_y] + wall_board[1][new_x, new_y] + \
                    pond_board[new_x, new_y] + castle_board[new_x, new_y] == 0 and \
                        masked[new_x, new_y] == True:
                    # from new_x, new_y, + 1 if linked to wall by diagonal, + 0.5 if linked to wall by vertical/horizontal
                    dx = [0, 0, 1, -1]
                    dy = [1, -1, 0, 0]
                    # Calculate linked wall score
                    cnt_wall = 0
                    cnt_pond = 0
                    cnt_bounder = 0
                    cnt_castle = 0
                    for j in range(4):
                        if new_x + dx[j] >= 0 and new_x + dx[j] < obs_size[0] and \
                            new_y + dy[j] >= 0 and new_y + dy[j] < obs_size[1]:
                            if wall_board[0][new_x + dx[j], new_y + dy[j]] == 1:
                                cnt_wall += 1
                            if new_x + dx[j] == 0 or new_x + dx[j] == obs_size[0] - 1 or \
                                new_y + dy[j] == 0 or new_y + dy[j] == obs_size[1] - 1:
                                cnt_bounder += 1
                            if pond_board[new_x + dx[j], new_y + dy[j]] == 1:
                                cnt_pond += 1
                            if castle_board[new_x + dx[j], new_y + dy[j]] == 1:
                                cnt_castle += 1
                                
                    score += (cnt_wall + cnt_bounder + cnt_pond + cnt_castle < 4) * cnt_wall * 0.5 - cnt_bounder * 0.25 - cnt_pond * 0.15 - cnt_castle * 0.05
                    
                    dx = [1, 1, -1, -1]
                    dy = [1, -1, 1, -1]
                    
                    for j in range(4):
                        if new_x + dx[j] >= 0 and new_x + dx[j] < obs_size[0] and \
                            new_y + dy[j] >= 0 and new_y + dy[j] < obs_size[1]:
                            if wall_board[0][new_x + dx[j], new_y + dy[j]] == 1:
                                score += 0.5
                    # +0.1 for future territory
                    if territory_board[0][new_x, new_y] == 1:
                        score += 0.2
                    
                    if territory_board[1][new_x, new_y] == 1:
                        score += 0.1 # need to damage territory's opponent in the future
                        
                scores[i] = score
            
        for j in range(len(self.action_space['Change'])):
            if valid_actions[self.action_map[('Change', self.action_space['Change'][j])]]:
                score = 1
                x, y = self.direction_map[self.action_space['Change'][j]]
                new_x, new_y = x + curr_x, y + curr_y
                
                dx = [0, 0, 1, -1]
                dy = [1, -1, 0, 0]
                for k in range(4):
                    if new_x + dx[k] >= 0 and new_x + dx[k] < obs_size[0] and \
                        new_y + dy[k] >= 0 and new_y + dy[k] < obs_size[1]:
                        if wall_board[0][new_x + dx[k], new_y + dy[k]] == 1:
                            score += 2.1
                                    
                dx = [1, 1, -1, -1]
                dy = [1, -1, 1, -1]
                
                for k in range(4):
                    if new_x + dx[k] >= 0 and new_x + dx[k] < obs_size[0] and \
                        new_y + dy[k] >= 0 and new_y + dy[k] < obs_size[1]:
                        if wall_board[0][new_x + dx[k], new_y + dy[k]] == 1:
                            score += 2.9
                                        
                scores[len(self.action_space['Move']) + j] = score
            
        scores[-1] = -0.9
        max_score = max(scores)
        max_score_actions = [i for i in range(len(scores)) if scores[i] == max_score]
        action = random.choice(max_score_actions)
        return action