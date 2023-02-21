import numpy as np


class Agent:

    def __init__(self, name, exp_rate=0.3):

        self.name = name
        self.states = []  # record all positions taken
        self.alpha = 0.2
        self.epsilon = exp_rate
        self.gamma = 0.9
        self.q_table = {}
        self.player_symbol = 0
        self.acc_reward = 0
        self.x = 0
        self.y = 0

    def getHash(self, env):
        boardHash = str(env.reshape(env.width * env.height))
        return boardHash

    # append a hash state
    def addState(self, state):
        self.states.append(state)

    def reset(self):
        self.x = 0
        self.y = 0
        self.states = []

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

        return action



    def train(self, rounds=100, env=None):

        for i in range(rounds):
            if i % 10 == 0:
                print("Rounds {}".format(i))
            while not self.isEnd:
                # Player 1
                positions = self.get_possible_actions()
                action = self.chooseAction(positions)
                # take action and upate board state
                self.updateState(action)
                board_hash = self.getHash()
                self.currentPlayer.addState(board_hash)
                # check board status if it is end

            if env.board[self.x, self.y] == -1:
                #Boom
                reward = -100
                self.reset()

            if self.pos() == env.end_position:
                reward = 100

    def pos(self):
        return  (self.x,self.y)

    def chooseAction(self, positions, current_board):

        estrategy = np.random.uniform(0, 1)

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

