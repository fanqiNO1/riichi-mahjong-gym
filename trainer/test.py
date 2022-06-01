import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.insert(0, parent)

from env.agent import AgentAgari
from env.mahjong import MahjongGame, MahjongEndGame
from trainer.greedy import Greedy
from env.player import Player
from env.ruleset import Ruleset

def main():
    game = MahjongGame(ruleset=Ruleset("../rules/ruleset.json"))
    for i in range(4):
        player = Player(f"Player{i}", is_manual=False, agent=Greedy(f"Player{i}"))
        game.set_player(i, player)

    game.play()


if __name__ == "__main__":
    main()