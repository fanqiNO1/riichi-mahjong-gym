from trainer.utils import encode
from env.agent import Agent
from env.utils import shanten_count
from env.deck import Deck
from env.tiles import Tile
from trainer.models.strategy import Replace

import numpy as np

class Greedy_DDPG(Agent):
    def __init__(self, name, args):
        super().__init__(name)
        self.replace = Replace(args)
        self.history = []
        self.a_loss = []
        self.c_loss = []

    def query(self, obs, action_space):
        for action in action_space:
            if action.action_type == "ron" or action.action_type == "tsumo":
                self.history.append({"obs": obs, "action": action})
                print(f"{self.name} chooses {action.action_type}")
                return action
        if obs["player_state"] == "passive":
            self.history.append({"obs": obs, "action": action})
            return action_space[0]
        else:
            # print(action_space)
            for action in action_space:
                if action.action_type == "reach":
                    print(f"{self.name} chooses {action.action_type}")
                    self.history.append({"obs": obs, "action": action})
                    return action
            logits = np.array(self.replace.get_logits(obs))
            hand = obs["hand"]
            shantens = []
            shantens.append(shanten_count(Deck(hand)))
            for i in range(len(hand)):
                hand_temp = Deck(hand) + obs["incoming_tile"] - hand[i]
                shantens.append(shanten_count(hand_temp))
            shantens = np.array(shantens)
            idx = np.argmin(shantens)
            actions = []
            tiles = []
            for i in range(len(shantens)):
                if shantens[i] == shantens[idx]:
                    action = action_space[i]
                    actions.append(action)
                    if action.action_type == "discard":
                        tiles.append(obs["incoming_tile"].get_34_id())
                    else:
                        tiles.append(Tile(int(action.action_string)).get_34_id())
            logit = logits[tiles]
            action_idx = np.argmax(logit)
            action = actions[action_idx]
            self.history.append({"obs": obs, "action": action, "action_dist": logits})
            return action
            # return action_space[idx]

    def update(self):
        a_loss, c_loss = self.replace.update()
        self.a_loss.append(a_loss)
        self.c_loss.append(c_loss)

    def save(self, name):
        self.replace.save(name)

    def load(self, name):
        self.replace.load(name)

    def replay_buffer_push(self, obs, action_dist, reward, next_obs, done):
        obs, action_dist, reward, next_obs, done = encode(
            obs, action_dist, reward, next_obs, done)
        self.replace.replay_buffer.push(obs, action_dist, reward, next_obs, done)