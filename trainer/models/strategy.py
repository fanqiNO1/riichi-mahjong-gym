from trainer.models.DDPG import DDPG

class Chii(DDPG):
    def __init__(self, args):
        super().__init__(obs_dim=170, hidden_dim=256, action_dim=2, args=args)
    
class Pon(DDPG):
    def __init__(self, args):
        super().__init__(obs_dim=170, hidden_dim=256, action_dim=2, args=args)

class Reach(DDPG):
    def __init__(self, args):
        super().__init__(obs_dim=170, hidden_dim=256, action_dim=2, args=args)

class Agari(DDPG):
    def __init__(self, args):
        super().__init__(obs_dim=170, hidden_dim=256, action_dim=2, args=args)

class Replace(DDPG):
    def __init__(self, args):
        super().__init__(obs_dim=170, hidden_dim=256, action_dim=34, args=args)