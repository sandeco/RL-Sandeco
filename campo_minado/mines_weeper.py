import numpy as np
import random

class Minesweeper:
    def __init__(self, size=10, num_mines=10):
        self.size = size
        self.num_mines = num_mines
        self.grid = np.zeros((size, size))
        self.grid[0, 0] = 1  # start point
        self.grid[size-1, size-1] = 2  # end point
        self.generate_mines()

        # Define a posição dos obstáculos no labirinto
    def generate_mines(self):
        positions = [(i, j) for i in range(self.size) for j in range(self.size)]
        positions.remove((0, 0))
        positions.remove((self.size - 1, self.size - 1))
        mines = random.sample(positions, self.num_mines)
        for mine in mines:
            self.grid[mine[0], mine[1]] = -1

    def showBoard(self):
        # p1: x  p2: o
        for i in range(0, self.size):
            print('-------------------------')
            aux = ""
            for j in range(0, self.size):
                aux = aux + " | " + str(self.grid[i][j])
            print(aux)
        print('-------------------------')

    def get_actions(self, state):
        actions = []

        row_min = max(0, state[0] - 1)
        row_max = min(self.size, state[0] + 2)
        col_min = max(0, state[1] - 1)
        col_max = min(self.size, state[1] + 2)

        range_row = range(row_min, row_max)
        range_col = range(col_min, col_max)

        for row in range_row:
            for col in range_col:
                if (row, col) == state:
                    continue
                actions.append((row, col))

        return actions

    def is_terminal(self, state):

        isTerminal = self.grid[state[0], state[1]] == 2

        return isTerminal


    def get_reward(self, state):
        if self.grid[state[0], state[1]] == -1:  # step on a mine
            return -100
        elif self.grid[state[0], state[1]] == 2:  # reach the end
            return 100
        else:
            return -1

    def getHash(self):
        boardHash = str(self.grid.reshape(self.size * self.size))
        return boardHash




