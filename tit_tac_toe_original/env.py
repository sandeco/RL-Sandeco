import numpy as np
import pickle


#https://raw.githubusercontent.com/MJeremy2017/reinforcement-learning-implementation/master/TicTacToe/ticTacToe.py

class Environment:
    def __init__(self, p1, p2):

        self.BOARD_ROWS = 3
        self.BOARD_COLS = 3

        self.board = np.zeros((self.BOARD_ROWS, self.BOARD_COLS))

        self.p1 = p1
        self.p1.player_symbol = 1

        self.p2 = p2
        self.p2.player_symbol = -1

        self.isEnd = False
        self.boardHash = None

        self.currentPlayer = self.p1
        self.first_player = self.currentPlayer

        self.reset()


    # get unique hash of current board state
    def getHash(self):
        self.boardHash = str(self.board.reshape(self.BOARD_COLS * self.BOARD_ROWS))
        return self.boardHash

    def winner(self):
        # row
        for i in range(self.BOARD_ROWS):
            if sum(self.board[i, :]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[i, :]) == -3:
                self.isEnd = True
                return -1
        # col
        for i in range(self.BOARD_COLS):
            if sum(self.board[:, i]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[:, i]) == -3:
                self.isEnd = True
                return -1
        # diagonal
        diag_sum1 = sum([self.board[i, i] for i in range(self.BOARD_COLS)])
        diag_sum2 = sum([self.board[i, self.BOARD_COLS - i - 1] for i in range(self.BOARD_COLS)])
        diag_sum = max(abs(diag_sum1), abs(diag_sum2))
        if diag_sum == 3:
            self.isEnd = True
            if diag_sum1 == 3 or diag_sum2 == 3:
                return 1
            else:
                return -1

        # tie
        # no available positions
        if len(self.get_possible_actions()) == 0:
            self.isEnd = True
            return 0
        # not end
        self.isEnd = False
        return None

    def get_possible_actions(self):
        positions = []
        for i in range(self.BOARD_ROWS):
            for j in range(self.BOARD_COLS):
                if self.board[i, j] == 0:
                    positions.append((i, j))  # need to be tuple
        return positions

    def updateState(self, position):
        self.board[position] = self.currentPlayer.player_symbol


    # only when game ends
    def giveReward(self):
        result = self.winner()
        # backpropagate reward
        if result == 1:
            self.p1.feedReward(1)
            self.p2.feedReward(0)
        elif result == -1:
            self.p1.feedReward(0)
            self.p2.feedReward(1)
        else:
            self.p1.feedReward(0.5)
            self.p2.feedReward(0.5)

    # board reset
    def reset(self):
        self.board = np.zeros((self.BOARD_ROWS, self.BOARD_COLS))
        self.boardHash = None
        self.isEnd = False

    def change_player(self):
        if self.currentPlayer.player_symbol == 1:
            self.currentPlayer = self.p2
        else:
            self.currentPlayer = self.p1

    def change_fisrt_player(self):
        if self.currentPlayer.player_symbol == 1:
            self.currentPlayer = self.p2
        else:
            self.currentPlayer = self.p1

    def train(self, rounds=100):
        for i in range(rounds):
            if i % 1000 == 0:
                print("Rounds {}".format(i))
            while not self.isEnd:
                # Player 1
                positions = self.get_possible_actions()
                action = self.currentPlayer.chooseAction(positions, self.board)
                # take action and upate board state
                self.updateState(action)
                board_hash = self.getHash()
                self.currentPlayer.addState(board_hash)
                # check board status if it is end

                win = self.winner()
                if win is not None:
                    # self.showBoard()
                    # ended with p1 either win or draw
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

                self.change_player()




    # play with human
    def play(self):

        while True:

            while not self.isEnd:
                # Player 1
                positions = self.get_possible_actions()
                action = self.currentPlayer.chooseAction(positions, self.board)
                # take action and upate board state
                self.updateState(action)
                self.showBoard()
                # check board status if it is end
                win = self.winner()
                if win is not None:
                    if win == 1:
                        print(self.p1.name, "wins!")
                    else:
                        print("tie!")
                    self.reset()
                    break

                self.change_player()

            self.change_fisrt_player()

    def showBoard(self):
        # p1: x  p2: o
        for i in range(0, self.BOARD_ROWS):
            print('-------------')
            out = '| '
            for j in range(0, self.BOARD_COLS):
                if self.board[i, j] == 1:
                    token = 'x'
                if self.board[i, j] == -1:
                    token = 'o'
                if self.board[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print('-------------')
