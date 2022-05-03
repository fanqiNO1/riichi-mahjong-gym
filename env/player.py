'''
File: player.py
Author: Kunologist
Description:
    This file contains a player, or agent, that can play the game of mahjong.
'''

import os
from env import action
from env.deck import Deck
from env.tiles import Tile
from env.action import Action
from env.agent import Agent
from env.utils import check_reach, check_agari, han_count, check_tenpai

def can_chii(tile_list, incoming_tile, obs):
    '''
    Function: can_chii(tile_list: `list`, incoming_tile: `Tile`) -> `bool`
    
    ## Description
    
    Checks whether a given tile can be chii.
    
    ## Parameters
    
    - `tile_list`: `list`
        The list of tiles.
    - `incoming_tile`: `Tile`
        The incoming tile.
    - `obs`: `dict`
        The observation of the game.
    
    ## Returns
    
    `None` or `list`
        `None` if no chii possible, a `list` of `Actions` if chii is possible.
    '''
    assert isinstance(tile_list, list)
    assert isinstance(incoming_tile, Tile)

    # Get the relationship between the player and the active player
    rel = (obs["active_player"] - obs["player_idx"]) % 4
    # rel == 1 means NEXT
    # rel == 2 means OPPOSING
    # rel == 3 means PREVIOUS
    # rel == 0 means SELF
    if rel == 0:
        print("[Warning] Checking chii for self.")
        return None
    if rel != 3:
        return None
    
    def create_chii_string(tile_chii, tile_hand_1, tile_hand_2):
        '''
        Function: create_chii_string(tile_chii: `Tile`, tile_hand_1: `Tile`, tile_hand_2: `Tile`, chii_rel: `int`) -> `str`
        
        ## Description
        
        Creates the chii string for a given tile.
        
        ## Parameters
        
        - `tile_chii`: `Tile`
            The tile to be chii.
        - `tile_hand_1`: `Tile`
            The first tile of the chii.
        - `tile_hand_2`: `Tile`
            The second tile of the chii.
        
        ## Returns
        
        `str`
            The chii string.
        '''
        assert isinstance(tile_chii, Tile)
        assert isinstance(tile_hand_1, Tile)
        assert isinstance(tile_hand_2, Tile)
        return "{}{}{}{}".format(
            "c",
            tile_chii.get_id(),
            tile_hand_1.get_id(),
            tile_hand_2.get_id()
        )

    # Consider red dora as non-red ones
    tile_list_red_dora_nullified = [Tile((tile.get_id() - 50) * 10 + 5) if tile.is_red_dora() else tile for tile in tile_list]

    if incoming_tile.get_suit() == "z":
        return None
    
    action_chii = []
    # cX1X2X3
    if incoming_tile.get_rank() <= 7:
        if Tile(incoming_tile.get_id() + 1) in tile_list_red_dora_nullified and Tile(incoming_tile.get_id() + 2) in tile_list_red_dora_nullified:
            # Check for hand red dora
            if Tile(incoming_tile.get_id() + 1) in tile_list and Tile(incoming_tile.get_id() + 2) in tile_list:
                # No red dora OK
                action_chii.append(Action.CHII(create_chii_string(incoming_tile, Tile(incoming_tile.get_id() + 1), Tile(incoming_tile.get_id() + 2))))
            # chii 3 hand 4 hand dora 5
            if incoming_tile == Tile(13) and Tile(14) in tile_list and Tile(51) in tile_list:
                action_chii.append(Action.CHII(create_chii_string(incoming_tile, Tile(14), Tile(51))))
            if incoming_tile == Tile(23) and Tile(24) in tile_list and Tile(52) in tile_list:
                action_chii.append(Action.CHII(create_chii_string(incoming_tile, Tile(24), Tile(52))))
            if incoming_tile == Tile(33) and Tile(34) in tile_list and Tile(53) in tile_list:
                action_chii.append(Action.CHII(create_chii_string(incoming_tile, Tile(34), Tile(53))))
            # chii 4 hand dora 5 hand 6
            if incoming_tile == Tile(14) and Tile(51) in tile_list and Tile(16) in tile_list:
                action_chii.append(Action.CHII(create_chii_string(incoming_tile, Tile(51), Tile(16))))
            if incoming_tile == Tile(24) and Tile(52) in tile_list and Tile(26) in tile_list:
                action_chii.append(Action.CHII(create_chii_string(incoming_tile, Tile(52), Tile(26))))
            if incoming_tile == Tile(34) and Tile(53) in tile_list and Tile(36) in tile_list:
                action_chii.append(Action.CHII(create_chii_string(incoming_tile, Tile(53), Tile(36))))
            
    # X1cX2X3
    if incoming_tile.get_rank() >= 2 and incoming_tile.get_rank() <= 8:
        if Tile(incoming_tile.get_id() - 1) in tile_list_red_dora_nullified and Tile(incoming_tile.get_id() + 1) in tile_list_red_dora_nullified:
            # Check for hand red dora
            if Tile(incoming_tile.get_id() - 1) in tile_list and Tile(incoming_tile.get_id() + 1) in tile_list:
                # No red dora OK
                action_chii.append(Action.CHII(create_chii_string(incoming_tile, Tile(incoming_tile.get_id() - 1), Tile(incoming_tile.get_id() + 1))))
            # hand 3 chii 4 hand dora 5
            if incoming_tile == Tile(14) and Tile(13) in tile_list and Tile(51) in tile_list:
                action_chii.append(Action.CHII(create_chii_string(incoming_tile, Tile(13), Tile(51))))
            if incoming_tile == Tile(24) and Tile(23) in tile_list and Tile(52) in tile_list:
                action_chii.append(Action.CHII(create_chii_string(incoming_tile, Tile(23), Tile(52))))
            if incoming_tile == Tile(34) and Tile(33) in tile_list and Tile(53) in tile_list:
                action_chii.append(Action.CHII(create_chii_string(incoming_tile, Tile(33), Tile(53))))
            # hand dora 5 chii hand 6 hand 7
            if incoming_tile == Tile(16) and Tile(51) in tile_list and Tile(17) in tile_list:
                action_chii.append(Action.CHII(create_chii_string(incoming_tile, Tile(51), Tile(17))))
            if incoming_tile == Tile(26) and Tile(52) in tile_list and Tile(27) in tile_list:
                action_chii.append(Action.CHII(create_chii_string(incoming_tile, Tile(52), Tile(27))))
            if incoming_tile == Tile(36) and Tile(53) in tile_list and Tile(37) in tile_list:
                action_chii.append(Action.CHII(create_chii_string(incoming_tile, Tile(53), Tile(37))))

    # X1X2cX3
    if incoming_tile.get_rank() >= 3:
        if Tile(incoming_tile.get_id() - 1) in tile_list_red_dora_nullified and Tile(incoming_tile.get_id() - 2) in tile_list_red_dora_nullified:
            # Check for hand red dora
            if Tile(incoming_tile.get_id() - 1) in tile_list and Tile(incoming_tile.get_id() - 2) in tile_list:
                # No red dora OK
                action_chii.append(Action.CHII(create_chii_string(incoming_tile, Tile(incoming_tile.get_id() - 1), Tile(incoming_tile.get_id() - 2))))
            # hand 3 hand 4 chii dora 5
            if incoming_tile == Tile(14) and Tile(13) in tile_list and Tile(51) in tile_list:
                action_chii.append(Action.CHII(create_chii_string(incoming_tile, Tile(13), Tile(51))))
            if incoming_tile == Tile(24) and Tile(23) in tile_list and Tile(52) in tile_list:
                action_chii.append(Action.CHII(create_chii_string(incoming_tile, Tile(23), Tile(52))))
            if incoming_tile == Tile(34) and Tile(33) in tile_list and Tile(53) in tile_list:
                action_chii.append(Action.CHII(create_chii_string(incoming_tile, Tile(33), Tile(53))))
            # hand dora 5 hand 6 chii 7
            if incoming_tile == Tile(17) and Tile(51) in tile_list and Tile(16) in tile_list:
                action_chii.append(Action.CHII(create_chii_string(incoming_tile, Tile(51), Tile(16))))
            if incoming_tile == Tile(27) and Tile(52) in tile_list and Tile(26) in tile_list:
                action_chii.append(Action.CHII(create_chii_string(incoming_tile, Tile(52), Tile(26))))
            if incoming_tile == Tile(37) and Tile(53) in tile_list and Tile(36) in tile_list:
                action_chii.append(Action.CHII(create_chii_string(incoming_tile, Tile(53), Tile(36))))

    if len(action_chii) > 0:
        return action_chii
    else:
        return None

class Player():
    '''
    Class: Player
    
    ## Description
    
    A class that represents a player, or agent, that can play the game of mahjong.
    '''
    
    name = None
    is_manual = None
    agent = None

    def __init__(self, name: str, is_manual: bool = False, agent: Agent = None):
        '''
        Constructor: __init__
        
        ## Description
        
        The constructor of the Agent class.
        
        ## Parameters
        
        - `name`: `str`
            The name of the player.
        - `is_manual`: `bool`
            Whether the player is a human or not.
        - `agent`: `Agent`
            The agent that performs actions for the player.
        '''
        assert isinstance(name, str)
        assert isinstance(is_manual, bool)
        self.name = name
        self.is_manual = is_manual
        if not is_manual:
            assert isinstance(agent, Agent)
            self.agent = agent
    
    def act(self, obs: dict):
        '''
        Method: act(obs)
        
        ## Description
        
        This method is called when the agent is supposed to act. The agent
        will be asked to act when the agent draws a tile, and when the agent
        has a chance to perform an action.
        
        ## Parameters
        
        - `obs`: `dict`
            The observation of the game.
        
        ## Returns

        `Action`
            The action that the agent wants to perform.
        '''
        # Get action space
        action_space = self.get_action_space(obs)
        # Query for action
        if self.is_manual:
            return self.manual_act(obs, action_space)
        else:
            return self.agent.query(obs, action_space)
    
    def manual_act(self, obs, action_space):
        '''
        Method: manual_act(obs, action_space)

        ## Description

        This method is called when the agent is supposed to act. The user
        will be asked to select an action.

        ## Parameters

        - `obs`: `dict`
            The observation of the game.
        - `action_space`: `list`
            All possible actions.
        '''
        if len(action_space) == 1:
            print("Skipping player P{}\n".format(obs["player_idx"]))
            return action_space[0]
        s = "You: P{} / Current: P{}\n\n".format(obs["player_idx"], obs["active_player"])
        print("You: P{} / Current: P{}\n\n".format(obs["player_idx"], obs["active_player"]))
        s += "Observation:\n"
        s += "Your Hand: " + obs["hand"].get_unicode_str() + " + " + obs["incoming_tile"].get_unicode_tile() + "\n"
        s += "Dora Indicators: " + Deck(obs["dora_indicators"]).get_unicode_str() + "\n\n"
        
        s += "Action space:\n"
        for i in range(len(action_space)):
            s += "{:02d}: {}\n".format(i, action_space[i].get_unicode_str())

        # Save to file
        with open("mahjong.hand.txt", "w", encoding="UTF-8") as f:
            f.write(s)
            print("Check the observation at " + os.path.abspath("mahjong.hand.txt"))
        
        # Get action from stdin
        action_id = None
        while action_id is None or action_id < 0 or action_id >= len(action_space):
            action_id = input("Select action: ")
            try:
                action_id = int(action_id)
            except:
                print("Wrong selection!")
                return None
        return action_space[action_id]

    def initialize(self):
        '''
        Method: initialize()
        
        ## Description

        This method is called when the agent is initialized.
        '''
        pass
    
    def get_action_space(self, obs: dict) -> list:
        '''
        Method: get_action_space()
        
        ## Description
        
        This method is called when the agent is supposed to get the action
        space.
        
        ## Parameters
        
        - `obs`: `dict`
            The observation of the game.

        ## Returns

        `list` of `Action`
            The action space of the agent. See `actions.md` documentation
            for more information.
        '''
  
        # Prepare action space
        action_space = []
        player_idx = obs["player_idx"]
        hand = obs["hand"].get_tiles().copy()
        calls = obs["calls"][player_idx]
        # Check if the player is active
        if obs["player_state"] == "active":

            # Default: allow discard
            action_space.append(Action.DISCARD())

            # Default: allow replace
            for tile in hand:
                action_space.append(Action.REPLACE(tile.get_id()))

            # Merge the incoming tile to the current hand
            hand.append(obs["incoming_tile"])
            hand.sort()
            # Player with incoming tile can call: kan, akan, discard, replace, reach, tsumo

            # Check for kan
            for call in calls:
                if call.find("p"):
                    # Pon found, check for kan
                    tile_id = int(call[-2:])
                    if Tile(tile_id) in hand:
                        action_space.append(Action.KAN(call))
            
            # Check for akan
            same_tile = 0
            previous_tile = Tile()
            for tile in hand:
                if tile == previous_tile:
                    same_tile += 1
                else:
                    same_tile = 0
                if same_tile == 4:
                    action_space.append(Action.AKAN(tile.get_id()))
                previous_tile = tile

            # Check for reach
            if obs["player_credit"][player_idx] >= 1000:
                reach_discard = check_reach(hand, calls)
                if reach_discard:
                    for reach_discard_tile in reach_discard:
                        # Check whether the tile is incoming
                        if reach_discard_tile == obs["incoming_tile"]:
                            action_space.append(Action.REACH(0))
                            # If also in hand, allow another reach action
                            if reach_discard_tile in obs["hand"].get_tiles():
                                action_space.append(Action.REACH(reach_discard_tile.get_id()))
                        else:
                            action_space.append(Action.REACH(reach_discard_tile.get_id()))
            
            # Check for tsumo
            if check_agari(hand, calls) and han_count(hand, calls) > 0:
                action_space.append(Action.TSUMO())
        
        elif obs["player_state"] == "end_game":

            # Player about to end game can call: ten, noten

            # Check for ten

            if check_tenpai(hand, calls):
                action_space.append(Action.TEN())

            # Always allow noten
            action_space.append(Action.NOTEN())

        elif obs["player_state"] == "passive":
            
            # Passive player can call: chii, pon, mkan, ron, noop
            
            action_space.append(Action.NOOP())

            incoming_tile = obs["incoming_tile"]

            # Check for chii

            can_chii_ret = can_chii(hand, incoming_tile, obs)
            if can_chii_ret is not None:
                action_space += can_chii_ret
                # action_space.append(Action.CHII(can_chii_ret))

            # Check for pon
            # TODO

            # Check for mkan
            # TODO

            # Check for ron
            hand.append(incoming_tile)
            if check_agari(hand, calls):
                action_space.append(Action.RON())
            
            # Check for chankan
            # TODO
        
        return action_space
