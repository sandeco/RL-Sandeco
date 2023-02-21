import numpy as np
import pickle

BOARD_COLS = 3
BOARD_ROWS = 3

class Player:
    def __init__(self, name, exp_rate=0.3):
        self.name = name
        self.states = []  # record all positions taken
        self.alpha = 0.2
        self.epsilon = exp_rate
        self.gamma = 0.9
        self.q_table = {}
        self.player_symbol = 0
        self.acc_reward = 0


    def incremente_reward(self, value):
        self.acc_reward = self.acc_reward + value

    def getHash(self, board):
        boardHash = str(board.reshape(BOARD_COLS * BOARD_ROWS))
        return boardHash

    def chooseAction(self, positions, current_board):

        estrategy =  np.random.uniform(0, 1)

        if estrategy <= self.epsilon:
            # take random action
            idx = np.random.choice(len(positions))
            action = positions[idx]
        else:
            value_max = -999
            for p in positions:
                next_board = current_board.copy()
                next_board[p] = self.player_symbol
                next_boardHash = self.getHash(next_board)
                value = 0 if self.q_table.get(next_boardHash) is None else self.q_table.get(next_boardHash)
                # print("value", value)
                if value >= value_max:
                    value_max = value
                    action = p
        # print("{} takes action {}".format(self.name, action))
        return action

    # append a hash state
    def addState(self, state):
        self.states.append(state)

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        for st in reversed(self.states):
            if self.q_table.get(st) is None:
                self.q_table[st] = 0
            self.q_table[st] += self.alpha * (self.gamma * reward - self.q_table[st])
            reward = self.q_table[st]

        self.incremente_reward(reward)

    def reset(self):
        self.states = []

    def savePolicy(self):
        fw = open('policy_agent', 'wb')
        pickle.dump(self.q_table, fw)
        fw.close()

    def loadPolicy(self, file):
        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()