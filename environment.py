import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
from utils import *

class TicTacToe3D(gym.Env):
    def __init__(self, N=5, K=5):
        super(TicTacToe3D, self).__init__()
        self.N = N
        self.K = K
        self.action_space = spaces.Discrete(N * N * N)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(N, N, N), dtype=np.int8)
        self.state = np.zeros((N, N, N), dtype=np.int8)
        # self.rewards = {}
        # self.terminalStates = []
        self.done = False

    def reset(self):
        self.state = np.zeros((self.N, self.N, self.N), dtype=np.int8)
        opponent_action = self.random_opponent_move()
        ox, oy, oz = np.unravel_index(opponent_action, (self.N, self.N, self.N))
        self.state[ox, oy, oz] = -1
        self.done = False
        return self.state

    def step(self, action):
        x, y, z = np.unravel_index(action, (self.N, self.N, self.N))
        if self.state[x, y, z] != 0:
            # Invalid move
            return self.state, -100, self.done, {}
        
        # Player always plays as 1, opponent as -1
        self.state[x, y, z] = 1
        
        # Check if player wins
        if check_winner(self.state, 1):
            self.done = True
            return self.state, 10, self.done, {}
        
        # Opponent's turn (random)
        if not self.done:
            flag = True
            ox = 0
            oy = 0
            oz = 0
            while flag is True:
                opponent_action = self.random_opponent_move()
                ox, oy, oz = np.unravel_index(opponent_action, (self.N, self.N, self.N))
                if self.state[ox, oy, oz] == 0:
                    flag=False

            self.state[ox, oy, oz] = -1
            
            # Check if opponent wins
            if check_winner(self.state, -1):
                self.done = True
                return self.state, -10, self.done, {}

        # Check for draw
        if not self.done and not np.any(self.state == 0):
            self.done = True
            return self.state, -2, self.done, {}
        
        return self.state, 0, self.done, {}
    

    def random_opponent_move(self):
        available_moves = np.argwhere(self.state == 0)
        if available_moves.size == 0:
            return -1  # No moves left
        move = np.random.choice(len(available_moves))
        action = np.ravel_multi_index(available_moves[move], (self.N, self.N, self.N))
        return action

    def render(self, mode='human'):
        # Use matplotlib or other libraries to visualize the 3D board
        render_board(self.state, self.N)
        # print(self.state)

# if __name__ == "__main__":
#     env = TicTacToe3D()
#     state = env.reset()
#     env.render()
#     action = env.action_space.sample()
#     state, reward, done, info = env.step(action)
#     env.render()
