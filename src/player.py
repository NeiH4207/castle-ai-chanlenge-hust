# Player class for the game
from src.agent import Agent


class Player(object):
    def __init__(self, ID, num_agents) -> None:
        super().__init__()
        self.ID = ID
        self.num_agents = num_agents
        self.init()
    
    @property
    def total_score(self):
        return self.area_score + self.wall_score
    
    def init(self):
        self.conquereds = {}
        self.castles = {}
        self.score = 0
        self.conquered_count = 0
        self.area_score = 0
        self.wall_score = 0
        self.agents = [Agent(i) for i in range(self.num_agents)]
    
    def set_agent_position(self, ID, position):
        """
        ID: int, position: tuple
        """
        self.agents[ID].set_pos(position)
        
    def update_scores(self, area_score, wall_score, treasure_score):
        self.area_score += area_score
        self.wall_score += wall_score
        self.treasure_score += treasure_score
        
    def reset_scores(self):
        self.area_score = 0
        self.wall_score = 0
        self.treasure_score = 0
        self.conquereds = {}
        self.castles = {}
        self.conquered_count = 0