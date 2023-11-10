import numpy as np

class RandomStep():
    def __init__(self, n_actions: int = 4, num_agents: int = 2) -> None:
        self.n_actions = n_actions
        self.num_agents = num_agents
        
    def get_action(self, state, epsilon=0.0):
        return np.random.randint(0, self.n_actions)