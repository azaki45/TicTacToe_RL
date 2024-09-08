# TicTacToe_RL
This repo contains code for the course project Introduction to Reinforcement Learning

## Problem Statement
The initial phase of the project was to build a 3D TicTacToe environment and train RL agents using Policy Iteration and Monte Carlo methods to play it. 

## Environment setup
The environment is set in a way that the opponents takes the first turn and plays random moves throughout the game.

The reward structure is as follows:
- +200 for winnning
- -200 for losing
- 0 for a draw
- -10 for a normal move (to incentivize the model to take as few moves to win as possible)

## Testing
To run it locally clone the repository and then follow the below steps

### Virtual environment setup
```
conda create -n tictactoe python=3
conda activate tictactoe
pip install numpy matplotlib gym
```

### Running the algorithms
The methods can be tried out in the following manner:
```
env = TicTacToe3D(N=3, K=3)
agent = MonteCarloAgent(env)
agent.train(num_episodes=1000)
```

## Note
For Policy Iteration, the state-space is too large and it is not feasible to run it. I have include the code to generate the valid states in utils.py but it is not recommended to run it unless there is sufficient time available. The states once generated will be stored in a JSON file and can be used for further experiments. 