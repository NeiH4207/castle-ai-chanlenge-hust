class Agent:
    def __init__(self, ID, name = None, symbol = None, color = None):
        self.ID = ID
        self.name = name
        self.symbol = symbol
        self.color = color
        self.last_move = None
        self.n_moves = 0
        self.x = None
        self.y = None
        
    def set_pos(self, pos):
        x, y = pos
        self.x = x
        self.y = y
        
    def get_pos(self):
        return (self.x, self.y)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return self.name == other.name