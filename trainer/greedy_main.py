import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.insert(0, parent)

import argparse
from env.agent import AgentAgari
from env.mahjong import MahjongGame, MahjongEndGame
from env.player import Player
from env.ruleset import Ruleset
from trainer.greedy_DDPG import Greedy_DDPG
from trainer.greedy import Greedy
from trainer.utils import get_reward, encode



def main(args):
    episode = 0
    greedy_ddpg = Greedy_DDPG(args.name, args)
    while episode < args.max_episodes:
        episode += 1
        print("Episode:", episode)
        step = 0
        game = MahjongGame(ruleset=Ruleset("../rules/ruleset.json"))
        for i in range(4):
            if i == args.player_idx:
                player = Player(args.name, is_manual=False, agent=greedy_ddpg)
            else:
                agent = Greedy(f"Player{i}")
                player = Player(f"Player{i}", is_manual=False, agent=agent)
            game.set_player(i, player)
        game.initialize_game()
        while True:
            if game.state["player_idx"] == args.player_idx:
                try:
                    game.step()
                except MahjongEndGame:
                    done = True
                    if episode % args.save_interval == 0:
                        greedy_ddpg.save(f"{args.name}_{episode}")
                        print(f"Saved model at episode {episode}")
                        with open("loss.txt", "a") as f:
                            for i in range(len(greedy_ddpg.a_loss)):
                                f.write(f"a_loss: {greedy_ddpg.a_loss[i]}, c_loss: {greedy_ddpg.c_loss[i]}\n")
                        greedy_ddpg.a_loss = []
                        greedy_ddpg.c_loss = []
                    break
                except TypeError:
                    done = True
                    if episode % args.save_interval == 0:
                        greedy_ddpg.save(f"{args.name}_{episode}")
                        print(f"Saved model at episode {episode}")
                        with open("loss.txt", "a") as f:
                            for i in range(len(greedy_ddpg.a_loss)):
                                f.write(f"a_loss: {greedy_ddpg.a_loss[i]}, c_loss: {greedy_ddpg.c_loss[i]}\n")
                        greedy_ddpg.a_loss = []
                        greedy_ddpg.c_loss = []
                    break
                obs, action = greedy_ddpg.history[-1]["obs"], greedy_ddpg.history[-1]["action"]
                next_obs = game.get_observation(args.player_idx)
                done = False if game.state["end_game"] == False else True
                if action.action_type == "replace" or action.action_type == "discard":
                    try:
                        reward = get_reward(obs, action, next_obs)
                        action_dist = greedy_ddpg.history[-1]["action_dist"]
                        greedy_ddpg.replay_buffer_push(
                            obs, action_dist, reward, next_obs, done)
                        greedy_ddpg.update()
                    except KeyError:
                        pass
                step += 1
                if args.episode_length <= step or done:
                    if episode % args.save_interval == 0:
                        greedy_ddpg.save(f"{args.name}_{episode}")
                        print(f"Saved model at episode {episode}")
                        with open("loss.txt", "a") as f:
                            for i in range(len(greedy_ddpg.a_loss)):
                                f.write(f"a_loss: {greedy_ddpg.a_loss[i]}, c_loss: {greedy_ddpg.c_loss[i]}\n")
                        greedy_ddpg.a_loss = []
                        greedy_ddpg.c_loss = []
                    break
            else:
                try:
                    game.step()
                except MahjongEndGame:
                    done = True
                    if episode % args.save_interval == 0:
                        greedy_ddpg.save(f"{args.name}_{episode}")
                        print(f"Saved model at episode {episode}")
                        with open("loss.txt", "a") as f:
                            for i in range(len(greedy_ddpg.a_loss)):
                                f.write(f"a_loss: {greedy_ddpg.a_loss[i]}, c_loss: {greedy_ddpg.c_loss[i]}\n")
                        greedy_ddpg.a_loss = []
                        greedy_ddpg.c_loss = []
                    break
                except TypeError:
                    done = True
                    if episode % args.save_interval == 0:
                        greedy_ddpg.save(f"{args.name}_{episode}")
                        print(f"Saved model at episode {episode}")
                        with open("loss.txt", "a") as f:
                            for i in range(len(greedy_ddpg.a_loss)):
                                f.write(f"a_loss: {greedy_ddpg.a_loss[i]}, c_loss: {greedy_ddpg.c_loss[i]}\n")
                        greedy_ddpg.a_loss = []
                        greedy_ddpg.c_loss = []
                    break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='greedy_DDPG')
    parser.add_argument('--max_episodes', default=50000, type=int)
    parser.add_argument('--episode_length', default=200, type=int)
    parser.add_argument('--save_interval', default=10000, type=int)
    parser.add_argument('--player_idx', default=0, type=int)

    parser.add_argument('--buffer_size', default=int(1e5), type=int)
    parser.add_argument('--tau', default=0.001, type=float)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--a_lr', default=0.0001, type=float)
    parser.add_argument('--c_lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epsilon', default=0.5, type=float)
    parser.add_argument('--epsilon_speed', default=0.99998, type=float)

    args = parser.parse_args()
    main(args)
