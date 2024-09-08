from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def check_winner(state, player, K=3):
    """
    Check if the given player has won in the given state.
    
    Args:
        state (np.ndarray): The current state of the N x N x N Tic-Tac-Toe board.
        player (int): The player to check (1 for the player, -1 for the opponent).
        K (int): The number of consecutive markers needed to win.

    Returns:
        bool: True if the player has won, otherwise False.
    """
    N = state.shape[0]

    # Check rows, columns, and depths
    for i in range(N):
        for j in range(N):
            # Check rows in 2D planes
            if all(state[i, j, :] == player):
                return True
            if all(state[i, :, j] == player):
                return True
            if all(state[:, i, j] == player):
                return True

    # Check 2D diagonals in each plane
    for i in range(N):
        # Diagonals in XY plane (fixed Z)
        if all(state[i, j, j] == player for j in range(N)):
            return True
        if all(state[i, j, N-j-1] == player for j in range(N)):
            return True

        # Diagonals in XZ plane (fixed Y)
        if all(state[j, i, j] == player for j in range(N)):
            return True
        if all(state[j, i, N-j-1] == player for j in range(N)):
            return True

        # Diagonals in YZ plane (fixed X)
        if all(state[j, j, i] == player for j in range(N)):
            return True
        if all(state[j, N-j-1, i] == player for j in range(N)):
            return True

    # Check 3D diagonals across the cube
    if all(state[i, i, i] == player for i in range(N)):
        return True
    if all(state[i, i, N-i-1] == player for i in range(N)):
        return True
    if all(state[i, N-i-1, i] == player for i in range(N)):
        return True
    if all(state[N-i-1, i, i] == player for i in range(N)):
        return True

    return False
    

def init_rewards(N=3, K=3):
    rewards = {}
    all_states = product([-1,0,1], repeat=N*N*N)

    for state in all_states:
        state_array = np.array(state).reshape((N,N,N))
        
        player_count = np.sum(state_array == 1)
        opponent_count = np.sum(state_array == -1)

        reward = 0

        if abs(player_count - opponent_count) > 1:
            continue
        elif check_winner(state, 1):
            reward = 200
        elif check_winner(state, -1):
            reward = -200
        else:
            continue

        rewards[tuple(state)] = reward

    return rewards

def rec1(i, j, k, state, all_states, N=3):
    if i >= N or j >= N or k >= N or i < 0 or j < 0 or k < 0:
        return
    
    count_p1 = np.sum(state==1)
    count_p2 = np.sum(state==-1)

    if abs(count_p1 - count_p2) > 1:
        return
    
    if tuple(state.flatten()) not in all_states:
        all_states.append(tuple(np.copy(state).flatten()))
    else:
        return
    
    # if(check_winner(state, 1) or check_winner(state, -1) or not np.any(state == 0)):
    #     return
    
    rec1(i+1, j, k, state, all_states, N)
    state[i,j,k] = 1
    rec1( i+1, j, k, state, all_states, N)
    state[i,j,k] = -1
    rec1(i+1, j, k, state, all_states, N)
    state[i,j,k] = 0

    rec1(i-1, j, k, state, all_states, N)
    state[i,j,k] = 1
    rec1( i-1, j, k, state, all_states, N)
    state[i,j,k] = -1
    rec1(i-1, j, k, state, all_states, N)
    state[i,j,k] = 0

    rec1(i, j+1, k, state, all_states, N)
    state[i,j,k] = 1
    rec1( i, j+1, k, state, all_states, N)
    state[i,j,k] = -1
    rec1(i, j+1, k, state, all_states, N)
    state[i,j,k] = 0

    rec1(i, j-1, k, state, all_states, N)
    state[i,j,k] = 1
    rec1( i, j-1, k, state, all_states, N)
    state[i,j,k] = -1
    rec1(i, j-1, k, state, all_states, N)
    state[i,j,k] = 0

    rec1(i, j, k+1, state, all_states, N)
    state[i,j,k] = 1
    rec1( i, j, k+1, state, all_states, N)
    state[i,j,k] = -1
    rec1(i, j, k+1, state, all_states, N)
    state[i,j,k] = 0

    rec1(i, j, k-1, state, all_states, N)
    state[i,j,k] = 1
    rec1( i, j, k-1, state, all_states, N)
    state[i,j,k] = -1
    rec1(i, j, k-1, state, all_states, N)
    state[i,j,k] = 0

    rec1(i+1, j+1, k, state, all_states, N)
    state[i,j,k] = 1
    rec1( i+1, j+1, k, state, all_states, N)
    state[i,j,k] = -1
    rec1(i+1, j+1, k, state, all_states, N)
    state[i,j,k] = 0

    rec1(i-1, j+1, k, state, all_states, N)
    state[i,j,k] = 1
    rec1( i-1, j+1, k, state, all_states, N)
    state[i,j,k] = -1
    rec1(i-1, j+1, k, state, all_states, N)
    state[i,j,k] = 0

    rec1(i+1, j-1, k, state, all_states, N)
    state[i,j,k] = 1
    rec1( i+1, j-1, k, state, all_states, N)
    state[i,j,k] = -1
    rec1(i+1, j-1, k, state, all_states, N)
    state[i,j,k] = 0

    rec1(i-1, j-1, k, state, all_states, N)
    state[i,j,k] = 1
    rec1( i-1, j-1, k, state, all_states, N)
    state[i,j,k] = -1
    rec1(i-1, j-1, k, state, all_states, N)
    state[i,j,k] = 0

    rec1(i, j+1, k+1, state, all_states, N)
    state[i,j,k] = 1
    rec1( i, j+1, k+1, state, all_states, N)
    state[i,j,k] = -1
    rec1(i, j+1, k+1, state, all_states, N)
    state[i,j,k] = 0

    rec1(i, j+1, k-1, state, all_states, N)
    state[i,j,k] = 1
    rec1( i, j+1, k-1, state, all_states, N)
    state[i,j,k] = -1
    rec1(i, j+1, k-1, state, all_states, N)
    state[i,j,k] = 0

    rec1(i, j-1, k+1, state, all_states, N)
    state[i,j,k] = 1
    rec1( i, j-1, k+1, state, all_states, N)
    state[i,j,k] = -1
    rec1(i, j-1, k+1, state, all_states, N)
    state[i,j,k] = 0

    rec1(i, j-1, k-1, state, all_states, N)
    state[i,j,k] = 1
    rec1( i, j-1, k-1, state, all_states, N)
    state[i,j,k] = -1
    rec1(i, j-1, k-1, state, all_states, N)
    state[i,j,k] = 0

    rec1(i+1, j, k+1, state, all_states, N)
    state[i,j,k] = 1
    rec1( i+1, j, k+1, state, all_states, N)
    state[i,j,k] = -1
    rec1(i+1, j, k+1, state, all_states, N)
    state[i,j,k] = 0

    rec1(i-1, j, k+1, state, all_states, N)
    state[i,j,k] = 1
    rec1( i-1, j, k+1, state, all_states, N)
    state[i,j,k] = -1
    rec1(i-1, j, k+1, state, all_states, N)
    state[i,j,k] = 0

    rec1(i+1, j, k-1, state, all_states, N)
    state[i,j,k] = 1
    rec1( i+1, j, k-1, state, all_states, N)
    state[i,j,k] = -1
    rec1(i+1, j, k-1, state, all_states, N)
    state[i,j,k] = 0

    rec1(i-1, j, k-1, state, all_states, N)
    state[i,j,k] = 1
    rec1( i-1, j, k-1, state, all_states, N)
    state[i,j,k] = -1
    rec1(i-1, j, k-1, state, all_states, N)
    state[i,j,k] = 0

    rec1(i+1, j+1, k+1, state, all_states, N)
    state[i,j,k] = 1
    rec1( i+1, j+1, k+1, state, all_states, N)
    state[i,j,k] = -1
    rec1(i+1, j+1, k+1, state, all_states, N)
    state[i,j,k] = 0

    rec1(i-1, j+1, k+1, state, all_states, N)
    state[i,j,k] = 1
    rec1( i-1, j+1, k+1, state, all_states, N)
    state[i,j,k] = -1
    rec1(i-1, j+1, k+1, state, all_states, N)
    state[i,j,k] = 0

    rec1(i+1, j-1, k+1, state, all_states, N)
    state[i,j,k] = 1
    rec1( i+1, j-1, k+1, state, all_states, N)
    state[i,j,k] = -1
    rec1(i+1, j-1, k+1, state, all_states, N)
    state[i,j,k] = 0

    rec1(i-1, j-1, k+1, state, all_states, N)
    state[i,j,k] = 1
    rec1( i-1, j-1, k+1, state, all_states, N)
    state[i,j,k] = -1
    rec1(i-1, j-1, k+1, state, all_states, N)
    state[i,j,k] = 0

    rec1(i+1, j+1, k-1, state, all_states, N)
    state[i,j,k] = 1
    rec1( i+1, j+1, k-1, state, all_states, N)
    state[i,j,k] = -1
    rec1(i+1, j+1, k-1, state, all_states, N)
    state[i,j,k] = 0

    rec1(i-1, j+1, k-1, state, all_states, N)
    state[i,j,k] = 1
    rec1( i-1, j+1, k-1, state, all_states, N)
    state[i,j,k] = -1
    rec1(i-1, j+1, k-1, state, all_states, N)
    state[i,j,k] = 0

    rec1(i+1, j-1, k-1, state, all_states, N)
    state[i,j,k] = 1
    rec1( i+1, j-1, k-1, state, all_states, N)
    state[i,j,k] = -1
    rec1(i+1, j-1, k-1, state, all_states, N)
    state[i,j,k] = 0

    rec1(i-1, j-1, k-1, state, all_states, N)
    state[i,j,k] = 1
    rec1( i-1, j-1, k-1, state, all_states, N)
    state[i,j,k] = -1
    rec1(i-1, j-1, k-1, state, all_states, N)
    state[i,j,k] = 0

    

def get_all_states(N=3):
    # all_states = []
    # state = np.zeros((N,N,N))
    all_states = []
    state = np.zeros((N,N,N))
    rec1(0,0,0,state,all_states,N)
    return all_states

def generate_configurations(board, player_count, opponent_count, turn, configurations):
    """
    Recursively generate valid 3x3x3 Tic-Tac-Toe configurations using backtracking.
    
    Args:
        board (np.ndarray): The current board state.
        player_count (int): Number of player markers on the board.
        opponent_count (int): Number of opponent markers on the board.
        turn (int): Current player's turn (1 for player, -1 for opponent).
        configurations (list): List to collect valid configurations.
    """
    # Check if the current board configuration is valid
    if abs(player_count - opponent_count) > 1:
        return

    # Check for terminal state
    if tuple(board.flatten()) not in configurations:
        print(f"state {len(configurations)}")
        configurations.append(tuple(np.copy(board).flatten()))
    else:
        return
    
    if check_winner(board, 1) or check_winner(board, -1) or not np.any(board == 0):
        return

    # Try placing the current player's marker in each empty cell
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if board[i, j, k] == 0:
                    # Place the marker
                    board[i, j, k] = turn
                    if turn == 1:
                        generate_configurations(board, player_count + 1, opponent_count, -turn, configurations)
                    else:
                        generate_configurations(board, player_count, opponent_count + 1, -turn, configurations)
                    # Backtrack
                    board[i, j, k] = 0

def generate_valid_configurations():
    """
    Generate all valid configurations for a 3x3x3 Tic-Tac-Toe board using backtracking.
    
    Returns:
        list: A list of valid 3x3x3 Tic-Tac-Toe configurations.
    """
    board = np.zeros((3, 3, 3), dtype=np.int8)
    configurations = []
    generate_configurations(board, 0, 0, -1, configurations)  # Start with player 2
    return configurations



def render_board(state):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    N = state.shape[0]
    for x in range(N):
        for y in range(N):
            for z in range(N):
                if state[x, y, z] == 1:
                    ax.scatter(x, y, z, c='r', marker='x')
                elif state[x, y, z] == -1:
                    ax.scatter(x, y, z, c='b', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

