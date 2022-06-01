from env.agent import Agent
from env.utils import shanten_count
from env.deck import Deck

import numpy as np

class Greedy(Agent):
    def __init__(self, name):
        super().__init__(name)

    def query(self, obs, action_space):
        for action in action_space:
            if action.action_type == "ron" or action.action_type == "tsumo":
                print(f"{self.name} chooses {action.action_type}")
                return action
        if obs["player_state"] == "passive":
            return action_space[0]
        else:
            # print(action_space)
            for action in action_space:
                if action.action_type == "reach":
                    print(f"{self.name} chooses {action.action_type}")
                    return action
            hand = obs["hand"]
            shantens = []
            shantens.append(shanten_count(Deck(hand)))
            for i in range(len(hand)):
                hand_temp = Deck(hand) + obs["incoming_tile"] - hand[i]
                shantens.append(shanten_count(hand_temp))
            # print(shantens)
            shantens = np.array(shantens)
            idx = np.argmin(shantens)
            return action_space[idx]