import numpy as np

class Minesweeper:
    def __init__(self):
        self.width = 10
        self.height = 10
        self.start_position = (0, 0)
        self.current_position = self.start_position
        self.end_position = (9, 9)
        self.obstacles = [(2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7),
                          (4, 2), (4, 7), (5, 2), (5, 7), (6, 2), (6, 7),
                          (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7)]

        # Cria uma matriz numpy para representar o labirinto
        self.board = np.zeros((self.width, self.height), dtype=int)

        # Define a posição de início e fim do labirinto
        self.board[self.end_position[0], self.end_position[1]] = 1

        # Define a posição dos obstáculos no labirinto
        for obstacle in self.obstacles:
            self.board[obstacle[0], obstacle[1]] = -1

    def showBoard(self):
        # p1: x  p2: o
        for i in range(0, self.width):
            print('-------------------------')
            aux = ""
            for j in range(0, self.height):
                aux = aux + " | " + str(self.board[i][j])
            print(aux)
        print('-------------------------')


    def possible_actions(self, agent):

        possible= []

        x = agent.x
        y = agent.y

        if (x-1)>=0:
            possible.append((x-1,y)) #left
        if (y-1)>=0:
            possible.append((x,y-1)) #top
        if (x+1)<=self.width:
            possible.append((x+1,y)) #right
        if (y+1<=self.height):
            possible.append((x,y+1)) #bottom

        return possible




