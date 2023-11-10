from collections import deque
from copy import deepcopy as dcopy
import random
import numpy as np
from board.screen import Screen
from src.player import Player
from src.state import State
import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

class AgentFighting(object):
    def __init__(self, args, configs, render = False):
        self.args = args
        self.configs = configs
        self._render = render
        
        self.action_space = {
            'Move': ['U', 'D', 'L', 'R', 'UL', 'UR', 'DL', 'DR'],
            'Change': ['U', 'D', 'L', 'R'],
            'Stay': 1
        }
        
        self.n_actions = len(self.action_space['Move']) + len(self.action_space['Change']) + 1
        self.num_players = 2
        self.screen = Screen(render=self._render)
        self.players = [Player(i, self.num_players) for i in range(self.num_players)]
        self.current_player = 0
        self.state = None
        self.last_diff_score = 0
        self.s_counter = {}
        self.reset()
        
    def render(self, state = None):
        if state is None:
            state = self.state
        if self._render:
            self.screen.load_state(state)
            self.screen.render()
    
    def save_image(self, path):
        if self._render:
            self.screen.save(path)
    
    def reset(self):
        """
        Resets the game by resetting player scores, creating a new map, and initializing the game state.
        :return: None
        """
        self.players[0].reset_scores()
        self.players[1].reset_scores()
        self.state = State(self.configs['map'], action_space=self.action_space)
        self.state.set_players(self.players)
        self.num_agents = self.state.num_agents
        self.state.make_random_map()
        if self._render:
            self.screen.init(self.state)
        self.num_agents = self.state.num_agents
    
    def in_bounds(self, coords):
        return 0 <= coords[0] < self.state.height and 0 <= coords[1] < self.state.width
    
    def is_valid_action(self, action):
        return action < self.n_actions
    
    def get_space_size(self):
        return self.get_state()['observation'].shape
            
    def get_state(self, partial=True, return_object=False):
        if return_object:
            return dcopy(self.state)
        else:
            return self.state.get_state(partial=partial)
        
        
    def hash_arr(self, arr: np.ndarray):
        s = ''.join([str(x) for x in arr.flatten()])
        return s
    
    def obs_string_representation(self, obs):
        """
        Returns a hash code for string representation of the state
        """
        s = self.hash_arr(obs)
        return hash(s)
    
    def is_visited_multiple_times(self, obs):
        return self.s_counter.get(self.obs_string_representation(obs), 0) > 1
    
    def is_terminal(self):
        """
        Checks if the game has ended by evaluating if there are any remaining turns left.

        :return: Boolean value indicating whether or not the game has ended.
        """
        return self.state.is_terminal()
            
    def get_winner(self):
        """
        Returns the winner of the game.

        :return: An integer representing the winner of the game.
        """
        if self.state.scores[0] > self.state.scores[1]:
            return 0
        elif self.state.scores[1] > self.state.scores[0]:
            return 1
        else:
            return -1
    
    def flip(self, matrix):
        return np.flip(matrix, axis=1)
    
    def rotate(self, matrix, k=1):
        return np.rot90(matrix, k=k)

    def get_symmetry_transition(self, state, action, next_state):
        flip = random.choice([True, False])
        action_type = self.state.get_type_action(action)
        if action_type[0] == 'Stay':
            return state, action, next_state
        
        if flip:
            direction = action_type[1]
            if action_type[0] == 'Move' or action_type[0] == 'Change':
                if direction == 'L':
                    direction = 'R'
                elif direction == 'R':
                    direction = 'L'
                elif direction == 'UL':
                    direction = 'UR'
                elif direction == 'UR':
                    direction = 'UL'
                elif direction == 'DL':
                    direction = 'DR'
                elif direction == 'DR':
                    direction = 'DL'
            
            action = self.state.action_map[(action_type[0], direction)]
                
            for i in range(state.shape[0]):
                state_layer = self.flip(state[i])
                state[i] = state_layer
                next_state_layer = self.flip(next_state[i])
                next_state[i] = next_state_layer
                
        action_type = self.state.get_type_action(action)
        k = random.choice([0, 1, 2, 3])
        
        for i in range(state.shape[0]):
            state_layer = self.rotate(state[i], k=k)
            state[i] = state_layer
            next_state_layer = self.rotate(next_state[i], k=k)
            next_state[i] = next_state_layer
            
        if action_type[0] == 'Move' or action_type[0] == 'Change':
            direction = action_type[1]
            for i in range(k):
                if direction == 'L':
                    direction = 'D'
                elif direction == 'R':
                    direction = 'U'
                elif direction == 'D':
                    direction = 'R'
                elif direction == 'U':
                    direction = 'L'
                elif direction == 'UL':
                    direction = 'DL'
                elif direction == 'DL':
                    direction = 'DR'
                elif direction == 'DR':
                    direction = 'UR'
                elif direction == 'UR':
                    direction = 'UL'
                    
                action = self.state.action_map[(action_type[0], direction)]
                    
            action = self.state.action_map[(action_type[0], direction)]
    
        return state, action, next_state
    
    def get_symmetric(self, obs, pi):
        obs = np.array(obs)
        pi = np.array(pi)
        sym_obs = []
        sym_pi = []
        for k in range(4):
            _obs = dcopy(obs)
            for i in range(obs.shape[0]):
                obs_layer = self.rotate(obs[i], k=k)
                _obs[i] = obs_layer
            pi_layer = dcopy(pi)
            for i in range(k):
                pi_layer = pi_layer[[2, 3, 1, 0, 6, 4, 7, 5, 10, 11, 9, 8, 12]]
            sym_obs.append(_obs)
            sym_pi.append(pi_layer)
            
        return sym_obs, sym_pi
        
    def get_valid_actions(self, state=None):
        valids = np.zeros(self.n_actions, dtype=bool)
        
        for action in range(self.n_actions):
            valids[action] = self.is_valid_action(action)
            
        return valids
    
    def get_last_diff_score(self):
        return self.last_diff_score
    
    def get_curr_agent_idx(self):
        return self.state.agent_current_idx
                    
    def get_diff_score(self):
        scores = self.state.scores
        curr_player_id = self.state.current_player
        return scores[curr_player_id] - scores[1 - curr_player_id]
                    
    def step(self, action, verbose=False):
        """
        This function performs a single step of the game by taking an action as input. The action 
        should be valid or else the function returns the reward. If the action is valid, then the 
        function updates the state of the game and returns the reward.

        Args:
            action: The action to be taken in the game.

        Returns:
            reward: The reward obtained from the step.
        """
        current_player = self.state.current_player
        previous_scores = self.state.scores
        diff_previous_scores = previous_scores[current_player] - previous_scores[1 - current_player]
        current_agent_idx = self.state.agent_current_idx
        
        self.state.next(action)
        
        if self._render:
            if self.state.agent_current_idx == 0:
                self.render(self.state)
            
        new_scores = self.state.scores
        diff_new_score = new_scores[current_player] - new_scores[1 - current_player]
        reward = 0.25 if diff_new_score > 0 else -0.5
        
        if diff_new_score > diff_previous_scores:
            reward += diff_new_score - diff_previous_scores
        elif diff_new_score < diff_previous_scores:
            reward -= diff_previous_scores - diff_new_score
        else:
            reward -= 0.1
            
        next_x, next_y = self.state.agent_coords_in_order[current_player][current_agent_idx]
        
        if self.state.territories[current_player][next_x][next_y] == 1:
            reward -= 0.25
        else:
            reward += 0.15
        
        if next_x == 0 or next_x == self.state.height - 1 or next_y == 0 or next_y == self.state.width - 1:
            reward -= 0.2
            
        self.last_diff_score = diff_new_score
        
        next_state = self.state.get_state()
        return next_state, reward, self.is_terminal()