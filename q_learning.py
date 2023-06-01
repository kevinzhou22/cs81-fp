#!/usr/bin/env python

# Author: Amit Das

import math
import random
import matplotlib.pyplot as plt

DECAY_RATE = 0.5
MAX_ENERGY = 20
ERRORS = [] # keep track of errors after each training iteration

class QLearning:
    """
    QLearning class implements the Q-learning algorithm

    Args:
        width (int)
        height(int)
        start_loc (tuple): (x, y)
        target_loc(tuple): (x, y)
    
    Attributes:
        alpha (float): learning rate
        gamma (float): discount factor
        q_table (dict): Q-table
        best_policy (dict): best action to take for each state
    """

    def __init__(self, width, height, start_loc, target_loc):
        self.target = target_loc
        self.alpha = 0.3
        self.gamma = 0.4
        self.q_table = {}
        self.best_policy = {}
        self.height = height
        self.width = width
        self.epsilon = 1
        self.is_close_to_robot = False
        self.in_energy_zone = False
        
        # initializing q_table with states, actions, and value of 0
        for i in range(width):
            for j in range(height):
                for e in range(MAX_ENERGY + 1):
                    self.q_table[(i, j, e, 'up')] = 0
                    self.q_table[(i, j, e, 'down')] = 0
                    self.q_table[(i, j, e, 'left')] = 0
                    self.q_table[(i, j, e, 'right')] = 0
        
    def update_q_table(self, state_action_pair):
        """
        Updates the Q-table for a given state-action pair;
        Mainly called by the training function

        Args:
            state_action_pair (tuple): (x, y, action)
        """
        
        state_x = state_action_pair[0]
        state_y = state_action_pair[1]
        state_energy = state_action_pair[2]
        action = state_action_pair[3]

        if action == 'up':
            new_state_x = state_x
            new_state_y = min(self.height - 1, state_y + 1)
        elif action == 'down':
            new_state_x = state_x
            new_state_y = max(0, state_y - 1)
        elif action == 'right':
            new_state_x = min(state_x + 1, self.width - 1)
            new_state_y = state_y
        elif action == 'left':
            new_state_x = max(state_x - 1, 0)
            new_state_y = state_y
        
        if new_state_x > 7.5 and new_state_y > 7.5:
            new_energy = MAX_ENERGY
        else:
            new_energy = max(0, state_energy - 1)
        
        next_state_max_val = max(self.q_table[(new_state_x, new_state_y, new_energy, 'right')],
                                 self.q_table[(new_state_x, new_state_y, new_energy, 'left')],
                                 self.q_table[(new_state_x, new_state_y, new_energy, 'up')],
                                 self.q_table[(new_state_x, new_state_y, new_energy, 'down')])
        

        q_value = (1-self.alpha)*self.q_table[state_action_pair] + self.alpha*(
            self.calculate_reward((new_state_x, new_state_y), new_energy) + self.gamma*next_state_max_val)
        
        self.q_table[state_action_pair] = q_value
        
    def calculate_reward(self, point, curr_energy):
        """
        Calculates the reward for moving to a new state

        Args:
            point (tuple): (x, y) location of the state
            curr_energy (int): current energy of robot
        
        Returns:
            float: reward at provided state
        """

        max_reward = 1000 # reward for closely following object

        # robot too close to object
        proximity_penalty = 0
        if self.is_close_to_robot:
            proximity_penalty = -100

        distance_to_object = math.sqrt((self.target[0] - point[0])**2 + (self.target[1] - point[1])**2)
        following_reward = max_reward/(1 + distance_to_object)
        
        low_energy_penalty = -1500 * math.sqrt((1 - (float(curr_energy) / 20)))

        edge_penalty = 0 # too close to edge of grid
        if point[0] >= 9 or point[1] >= 9 or point[0] <= 1 or point[1] <= 1:
            edge_penalty = -1000

        return following_reward + proximity_penalty + low_energy_penalty + edge_penalty
    
    def training(self, iterations = 5):
        """
        Performs the Q-learning training process

        Args:
            iterations (int): number of iterations to train
        """
        
        for i in range(iterations):
            prev_total = sum(self.q_table.values())
            for key in self.q_table.keys():
                self.update_q_table(key)
            self.get_best_policy(self.q_table)
            new_total = sum(self.q_table.values())

            ERRORS.append(abs(new_total - prev_total))
            if abs(new_total - prev_total) < 100:
                print((new_total), (prev_total))
                print("Converges during Iteration " + str(i))
                break # stops training after convergence

    
    def get_best_policy(self, q_table):
        """
        Determines best action for each state based on the Q-table

        Args:
            q_table (dict): The Q-table after training
        """

        all_states = set()
        for key in q_table.keys():
            all_states.add((key[0], key[1], key[2]))

        self.epsilon = self.epsilon * DECAY_RATE
        
        
        for state in all_states:
            actions = {
                "right": self.q_table[(state[0], state[1], state[2], "right")],
                "left": self.q_table[(state[0], state[1], state[2], "left")],
                "up": self.q_table[(state[0], state[1], state[2], "up")],
                "down": self.q_table[(state[0], state[1], state[2], "down")]
            }

            # Implementation of epsilon-greedy policy
            random_val = random.random()
            if random_val > self.epsilon:
                self.best_policy[(state[0], state[1], state[2])] = max(actions, key = actions.get)
            else:
                pos_actions = ["right", "left", "up", "down"]
                random_index = random.randint(0, 3)
                self.best_policy[(state[0], state[1], state[2])] = pos_actions[random_index]

if __name__ == "__main__":
    q_learning = QLearning(20, 20, (0, 0), (5, 5))
    q_learning.training(200)
    q_learning.get_best_policy(q_learning.q_table)

    # Plotting the error over iterations graph
    plt.plot([i for i in range(len(ERRORS))], ERRORS, color='green', linewidth=2, marker='o', markersize=4)

    # Setting up the grid
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.ylabel("Error")
    plt.xlabel("Iterations")
    plt.title("Error over Iterations")

    # Setting up the axes
    plt.axhline(y=0, color='black', linewidth=1)
    plt.axvline(x=0, color='black', linewidth=1)
    plt.show()

