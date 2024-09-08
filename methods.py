import numpy as np
from collections import defaultdict
from utils import *
import random

class PolicyIterationAgent:
    def __init__(self, env, discount_factor=0.99, theta=0.0001):
        self.env = env
        # self.bins = bins
        self.discount_factor = discount_factor
        self.theta = theta
        self.space_size = env.N * env.N * env.N
        self.policy = defaultdict(lambda: np.random.randint(0,self.space_size,self.space_size) / (self.space_size))  # Random initial policy
        self.value_function = defaultdict(float)
        self.visited_states = []  # Track visited states
        self.final_states = []  # Track final states
        self.all_states = []
        self.final_rewards = []


    def compute_value(self, state):
        state_value = 0.0
        if self.env.is_terminal(np.array(state).reshape((self.env.N, self.env.N, self.env.N))):
            return state_value
        action_probabilities = self.policy[state]
        for action, action_prob in enumerate(action_probabilities):
            if(action_prob == 0):
                continue
            next_state, reward, done, _ = self.env.simulate_step(state, action)  # Simulate action
            if(reward == -100):
                continue
            state_value += action_prob * (reward + self.discount_factor * self.value_function[tuple(next_state.flatten())])

        return state_value

    def compute_best_action(self, state):
        best_value = float('-inf')
        best_action = None
        if self.env.is_terminal(np.array(state).reshape((self.env.N, self.env.N, self.env.N))):
            return best_action
        for action in range(self.env.action_space.n):
            next_state, reward, done, _ = self.env.simulate_step(state, action)  # Simulate action
            if(reward == -100):
                continue

            action_value = reward + self.discount_factor * self.value_function[tuple(next_state.flatten())]
            if action_value > best_value:
                best_value = action_value
                best_action = action
        return best_action

    def policy_evaluation(self):

        while True:
            delta = 0
            for state in self.all_states:  # Only iterate over visited states
                old_value = self.value_function[state]
                new_value = self.compute_value(state)
                self.value_function[state] = new_value
                delta = max(delta, abs(old_value - new_value))
            if delta < self.theta:
                break

    def policy_improvement(self):
        for state in self.all_states:  
            best_action = self.compute_best_action(state)
            if(best_action == None):
                continue
            self.policy[state] = np.zeros(self.env.action_space.n)
            self.policy[state][best_action] = 1.0 # updating the best action probability to 1

    def train(self, num_episodes):
        for episode in range(num_episodes):
            print(f"Episode {episode}")
            print("------------------------------------------")
            self.final_states = []
            state = self.env.reset()
            done = False
            while not done:
                if tuple(state.flatten()) not in self.visited_states:
                    self.visited_states.append(tuple(state.flatten()))  # Add state to visited set
                    print("Vistied a new state")
                    self.policy[tuple(state.flatten())] = np.random.randint(0,self.space_size,self.space_size) / (self.space_size)
                if tuple(state.flatten()) not in self.final_states:
                    self.final_states.append(tuple(state.flatten()))  # Add state to final set
                
                while True:
                    action = np.argmax(self.policy[tuple(state.flatten())])
                    next_state, reward, done, _ = self.env.step(action)
                    if reward!=-100:
                        break
                    else:
                        self.policy[tuple(state.flatten())] = np.random.randint(0,self.space_size,self.space_size) / (self.space_size)
                print(f"Action taken = {action}")

                state = next_state
        

            self.policy_evaluation()
            r=0
            gam = self.discount_factor
            i = 0
            for st in self.final_states:
                r += pow(gam,i)*self.value_function[st]
                i+=1

            self.policy_improvement()

            self.final_rewards.append(r/len(self.final_states))

        print("Training done")
    
    def train2(self, num_episodes):

        for episode in range(num_episodes):
            print(f"Episode {episode+1}")
            print("------------------------------------------")
            # state = self.env.reset()
            self.policy_evaluation()
            r=0
            gam = self.discount_factor
            i = 0
            for st in self.all_states:
                r += pow(gam,i)*self.value_function[st]
                # r += self.value_function[st]
                i+=1

            # if self.policy_improvement():
            #     self.final_rewards.append(r)
            #     break
            self.policy_improvement()
            self.final_rewards.append(r)

        print("Training done")

            
class MonteCarloAgent:
    def __init__(self, env, discount_factor=0.9, epsilon=1):
        self.env = env
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.min_epsilon = 0.01
        self.decay_factor = 0.995
        self.policy = defaultdict(lambda: np.ones(self.env.action_space.n) / self.env.action_space.n)  # Epsilon-greedy initial policy
        self.value_function = defaultdict(float)
        self.returns = defaultdict(list)  # To store returns for each state-action pair
        self.final_rewards = []
        self.final_q = []
        self.visited = defaultdict(float)
        self.alpha = 0.1

    def select_action(self, state):
        """
        Select an action using an epsilon-greedy policy.
        """
        state_key = tuple(state.flatten())
        if random.random() < self.epsilon:
            # Explore: select a random action
            return self.env.action_space.sample()
        else:
            # Exploit: select the best action based on the current policy
            return np.argmax(self.policy[state_key])

    def generate_episode(self):
        """
        Generate an episode following the current policy.
        
        Returns:
            episode (list): A list of (state, action, reward) tuples.
        """
        episode = []
        self.visited_states = []
        state = self.env.reset()
        done = False

        for i in range(200):
          action = self.select_action(state)
          next_state, reward, done, _ = self.env.step(action)
          
          # Store the state, action, and reward
          episode.append((tuple(state.flatten()), action, reward))
          self.visited_states.append(tuple(state.flatten()))
          if done:
            break
          state = next_state
        
        self.epsilon = self.epsilon*self.decay_factor
        self.epsilon = max(self.min_epsilon, self.epsilon)
        return episode

    def update_policy(self, episode):
        """
        Update the policy and value function based on the returns observed in the episode using Every-Visit Monte Carlo.
        """
        G = 0  # Initialize the return
        # Traverse the episode backwards
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            self.visited[state]+=1
            G = self.discount_factor * G + reward
            

            self.returns[(state, action)].append(G)

            self.value_function[state] += self.alpha*(np.mean(self.returns[(state,action)]) - self.value_function[state])
            
            # Improve policy to take the action with the highest return
            best_action = None
            best_value = float('-inf')
            for a in range (self.env.action_space.n):
              next_state, rew, done, _ = self.env.simulate_step(state,a)
              if(reward==-100):
                continue
              action_value = rew + self.discount_factor * self.value_function[tuple(next_state.flatten())]
              if action_value > best_value:
                  best_value = action_value
                  best_action = a
            self.policy[state] = np.zeros(self.env.action_space.n)
            self.policy[state][best_action] = 1.0 # updating the best action probability to 1


    def train(self, num_episodes):
        """
        Train the agent using Every-Visit Monte Carlo learning.
        """
        for episode_num in range(num_episodes):
            print(f"Episode {episode_num + 1}")
            episode = self.generate_episode()
            self.update_policy(episode)

            r=0
            i=0
            for st,_,__ in episode:
              r=pow(self.discount_factor,i)*self.value_function[st]
              i+=1
            
            # if (episode_num + 1)%100 == 0:
            self.final_rewards.append(r)
            # print(f"Visited states: {len(self.returns)}")
