from env.agent import Agent
from trainer.models.strategy import Chii, Pon, Agari, Replace
from trainer.utils import encode

class Ichihime(Agent):
    def __init__(self, name, args):
        super().__init__(name)
        self.args = args
        self.history = []
        self.chii = Chii(args)
        self.pon = Pon(args)
        self.agari = Agari(args)
        self.replace = Replace(args)

    def query(self, obs, action_space):
        action_chosen = None
        can_replace = False
        action_dist = None
        for action in action_space:
            if action.action_type == "ron" or action.action_type == "tsumo":
                is_agari, action_dist = self.agari.choose_action(obs)
                if is_agari:
                    action_chosen = action
                    break
            elif action.action_type == "chi":
                is_chii, action_dist = self.chii.choose_action(obs)
                if is_chii:
                    action_chosen = action
                    break
            elif action.action_type == "pon":
                is_pon, action_dist = self.pon.choose_action(obs)
                if is_pon:
                    action_chosen = action
                    break
            elif action.action_type == "replace" or action.action_type == "discard":
                can_replace = True
        if action_chosen is None:
            if can_replace:
                action, action_dist = self.replace.choose_action(obs, action_space)
                action_chosen = action_space[action]
            else:
                action_chosen = action_space[0]
        
        self.history.append(
            {"obs": obs, "action": action_chosen, "action_space": action_space, "action_dist": action_dist})
        return action_chosen

    def update(self):
        self.chii.update()
        self.pon.update()
        self.agari.update()
        self.replace.update()

    def save(self, name):
        self.chii.save(f"{name}_chii")
        self.pon.save(f"{name}_pon")
        self.agari.save(f"{name}_agari")
        self.replace.save(f"{name}_replace")

    def replay_buffer_push(self, obs, action, action_dist, reward, next_obs, done):
        if action.action_type == "chii":
            obs, action_dist, reward, next_obs, done = encode(
                    obs, action_dist, reward, next_obs, done)
            self.chii.replay_buffer.push(obs, action_dist, reward, next_obs, done)
        elif action.action_type == "pon":
            obs, action_dist, reward, next_obs, done = encode(
                    obs, action_dist, reward, next_obs, done)
            self.pon.replay_buffer.push(obs, action_dist, reward, next_obs, done)
        elif action.action_type == "ron" or action.action_type == "tsumo":
            obs, action_dist, reward, next_obs, done = encode(
                    obs, action_dist, reward, next_obs, done)
            self.agari.replay_buffer.push(obs, action_dist, reward, next_obs, done)
        else:
            obs, action_dist, reward, next_obs, done = encode(
                    obs, action_dist, reward, next_obs, done)
            self.replace.replay_buffer.push(
                obs, action_dist, reward, next_obs, done)
