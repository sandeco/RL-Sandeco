import random
import numpy as np


class Agent:

    def __init__(self, num_states, num_actions=8, learning_rate=0.5, discount_factor=0.9, exploration_rate=0.1):

        self.states = []  # record all positions taken
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

        #para cada estado pode-se tomar 8 ações possíveis
        self.q_table = np.zeros((num_states, num_actions))


    def choose_action(self, state, possible_actions):

        estrategy =  np.random.uniform(0, 1)

        if estrategy <= self.exploration_rate:
            # take random action
            idx = np.random.choice(len(possible_actions))
            action = possible_actions[idx]
        else:
            values = self.q_table[state]
            max_value = np.max(values)
            #indices = [i for i in range(len(actions)) if values[i] == max_value]
            indices = []

            for i in range(len(possible_actions)):
                if values[i] == max_value:
                    indices.append(i)

            index = random.choice(indices)
            return possible_actions[index]

        return action

    def update_q_table(self, state, action, reward, next_state):
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        #new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * next_max)

        temp1 = 1 - self.learning_rate
        temp2 = old_value * temp1
        temp3 = reward + self.discount_factor * next_max
        temp4 = self.learning_rate * temp3
        new_value = temp2 + temp4

        self.q_table[state][action] = new_value

    def pos(self):
        return  (self.x,self.y)


