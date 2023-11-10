"""
Created on Tue Apr 27 2:19:47 2023
@author: hien
"""
from __future__ import division
import json
import logging
import time
from algorithms.RandomStep import RandomStep
from algorithms.StupidMove import StupidMove
from src.environment import AgentFighting
from src.utils import set_seed
log = logging.getLogger(__name__)
from argparse import ArgumentParser

def argument_parser():
    parser = ArgumentParser()
    parser.add_argument('--render', action='store_true', default=True,
                        help='Whether to render the game')
    return parser.parse_args()

def main():
    args = argument_parser()
    configs = json.load(open('configs/map.json'))
    env = AgentFighting(args, configs, args.render)
    set_seed(0)
    observation_shape = env.get_space_size()
    n_actions = env.n_actions
    logging.info('Observation shape: {}'.format(observation_shape))
    logging.info('Number of actions: {}'.format(n_actions))
    player_brain_1 = RandomStep(n_actions=env.n_actions, num_agents=env.num_agents)
    player_brain_2 = StupidMove(n_actions=env.n_actions, num_agents=env.num_agents)
    """
    partial = True (default) if you want to get the partial state,
    the environment will return the a matrix of size (obs_range x 2 + 1) x (obs_range x 2 + 1) 
    cropped from the full state with the center cell being the current agent.
    obs_range is set in configs/map.json file
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
    # """
    # state = env.get_state(partial=False)
    state = env.get_state()
    
    while not env.is_terminal():
        if state['player-id'] == 0:
            action = player_brain_1.get_action(state)
        else:
            action = player_brain_2.get_action(state)
        log.info('PlayerID: {}, AgentID: {}, Action: {}'.format(state['player-id'], state['current-agent-id'], action))
        next_state, reward, done = env.step(action, verbose=True)
        state = next_state
        env.render()
        time.sleep(0.05)
    
    winner = env.get_winner()
    if winner == -1:
        logging.info('Game ended. Draw')
    else:
        logging.info('Game ended. Winner: {}'.format(env.get_winner()))

if __name__ == "__main__":
    main()