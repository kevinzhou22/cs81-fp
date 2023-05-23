#!/usr/bin/env python

import math
import random

DECAY_RATE = 0.5

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
        self.alpha = 0.2
        self.gamma = 0.4
        self.q_table = {}
        self.best_policy = {}
        self.height = height
        self.width = width
        self.epsilon = 1
        
        # initializing q_table with states, actions, and value of 0
        for i in range(width):
            for j in range(height):
                self.q_table[(i, j, 'up')] = 0
                self.q_table[(i, j, 'down')] = 0
                self.q_table[(i, j, 'left')] = 0
                self.q_table[(i, j, 'right')] = 0
        
    def update_q_table(self, state_action_pair):
        """
        Updates the Q-table for a given state-action pair;
        Mainly called by the training function

        Args:
            state_action_pair (tuple): (x, y, action)
        """
        
        state_x = state_action_pair[0]
        state_y = state_action_pair[1]
        action = state_action_pair[2]

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
        
        next_state_max_val = max(self.q_table[(new_state_x, new_state_y, 'right')],
                                 self.q_table[(new_state_x, new_state_y, 'left')],
                                 self.q_table[(new_state_x, new_state_y, 'up')],
                                 self.q_table[(new_state_x, new_state_y, 'down')])
        

        q_value = (1-self.alpha)*self.q_table[state_action_pair] + self.alpha*(
            self.calculate_reward((new_state_x, new_state_y)) + self.gamma*next_state_max_val)
        
        self.q_table[state_action_pair] = q_value
        
    def calculate_reward(self, point):
        """
        Calculates the reward for moving to a new state

        Args:
            point (tuple): (x, y) location of the state
        
        Returns:
            float: reward at provided state
        """
        # bonus = 0
        # if point[0] > 8 or point[0] < 2:
        #     if point[1] > 8 or point[1] < 2:
        #         bonus = 5

        return -math.sqrt((self.target[0] - point[0])**2 + (self.target[1] - point[1])**2)
    
    def training(self, iterations = 5):
        """
        Performs the Q-learning training process

        Args:
            iterations (int): number of iterations to train
        """
        for i in range(iterations):
            for key in self.q_table.keys():
                self.update_q_table(key)
            self.get_best_policy(self.q_table)
    
    def get_best_policy(self, q_table):
        """
        Determines best action for each state based on the Q-table

        Args:
            q_table (dict): The Q-table after training
        """
        all_states = set()
        for key in q_table.keys():
            all_states.add((key[0], key[1]))
        
        self.epsilon = self.epsilon * DECAY_RATE
        
        for state in all_states:
            actions = {
                "right": self.q_table[(state[0], state[1], "right")],
                "left": self.q_table[(state[0], state[1], "left")],
                "up": self.q_table[(state[0], state[1], "up")],
                "down": self.q_table[(state[0], state[1], "down")]
            }

            # Implementation of epsilon-greedy policy
            random_val = random.random()
            if random_val > self.epsilon:
                self.best_policy[(state[0], state[1])] = max(actions, key = actions.get)
            else:
                pos_actions = ["right", "left", "up", "down"]
                random_index = random.randint(0, 3)
                self.best_policy[(state[0], state[1])] = pos_actions[random_index]

if __name__ == "__main__":
    
    q_learning = QLearning(20, 20, (0, 0), (5, 5))
    q_learning.training(10)
    q_learning.get_best_policy(q_learning.q_table)

    print(q_learning.best_policy)
