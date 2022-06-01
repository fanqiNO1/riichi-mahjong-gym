# trainer

Just a naive agent of the mahjong. Some ideas come from [Suphx](https://arxiv.org/abs/2003.13590).

And considering that in the game of Majsoul, Ichihime is our good companion (for sure), so we named our agent Ichihime.

## Baseline

You should prepare the environment first.

`cd trainer && pip install -r requirements.txt`

Then you can train the agent.

`python main.py`

You can get the help by running `python main.py -h` to set the parameters.

### Architecture

There are four main models in the agent, which are `Chii`, `Pon`, `Agari`, and `Replace`.

The models are mainly based on the [DDPG](https://arxiv.org/abs/1509.02971) paper.

## Greedy-DDPG

You should prepare the environment first.

`cd trainer && pip install -r requirements.txt`

Then you can train the agent.

`python greedy_main.py`

You can get the help by running `python greedy_main.py -h` to set the parameters.

### Architecture

Just a combination of `DDPG` and `Greedy`. :smile: