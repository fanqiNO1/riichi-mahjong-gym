import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.insert(0, parent)

from env.utils import shanten_count
from env.deck import Deck
import numpy as np


def encode(obs, action, reward, next_obs, done):
    obs = encode_obs(obs)
    next_obs = encode_obs(next_obs)
    action = encode_action(action)
    reward = encode_reward(reward)
    done = encode_done(done)
    return obs, action, reward, next_obs, done


def encode_obs(obs):
    hand = np.array(Deck(obs["hand"]).get_34_array())
    discarded_tiles_0 = np.array(
        Deck(obs["discarded_tiles"][0]).get_34_array())
    discarded_tiles_1 = np.array(
        Deck(obs["discarded_tiles"][1]).get_34_array())
    discarded_tiles_2 = np.array(
        Deck(obs["discarded_tiles"][2]).get_34_array())
    discarded_tiles_3 = np.array(
        Deck(obs["discarded_tiles"][3]).get_34_array())
    obs = np.concatenate((hand, discarded_tiles_0, discarded_tiles_1, discarded_tiles_2, discarded_tiles_3)).astype(np.float32)
    return obs


def encode_action(action):
    action = np.array(action)
    return action


def encode_reward(reward):
    return np.array([reward]).astype(np.float32)


def encode_done(done):
    if done == False:
        return np.array([0]).astype(np.float32)
    else:
        return np.array([1]).astype(np.float32)


def get_reward(obs, action, next_obs):
    reward = 0
    if action.action_type == "ron" or action.action_type == "tsumo" or action.action_type == "reach":
        reward = get_reward_agari(obs, action, next_obs)
    elif action.action_type == "chii" or action.action_type == "pon":
        reward = get_reward_call(obs, action, next_obs)
    else:
        reward = get_reward_replace(obs, action, next_obs)
    return reward


def get_reward_agari(obs, action, next_obs):
    player_idx = obs["player_idx"]
    reward = next_obs["credits"][player_idx] - obs["credits"][player_idx]
    reward /= 1000
    return reward


def get_reward_call(obs, action, next_obs):
    current_shanten = shanten_count(Deck(obs["hand"]))
    min_shanten = 6
    for i in next_obs["hand"]:
        temp_hand = Deck(next_obs["hand"]) - i
        temp_shanten = shanten_count(temp_hand)
        min_shanten = min(min_shanten, temp_shanten)
    reward = current_shanten - min_shanten
    return reward


def get_reward_replace(obs, action, next_obs):
    current_shanten = shanten_count(Deck(obs["hand"]))
    next_shanten = shanten_count(Deck(next_obs["hand"]))
    reward = current_shanten - next_shanten
    return reward

def get_reward_greedy(obs, action, next_obs):
    reward = get_reward_replace(obs, action, next_obs)
    reward += get_reward_agari(obs, action, next_obs)
    return reward
