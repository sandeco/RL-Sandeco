import numpy as np
import random

# Definir o jogo da velha (iic-tac-toe)

# Classe para o ambiente do jogo
class TicTacToeEnvironment:
    def __init__(self):
        self.X = 1
        self.O = -1
        self.EMPTY = 0
        self.BOARD_ROWS = 3
        self.BOARD_COLS = 3

        self.board = np.zeros((self.BOARD_ROWS, self.BOARD_COLS))
        self.current_player = self.X
        self.winner = None
        self.ended = False

    # Reiniciar o ambiente
    def reset(self):
        self.board = np.zeros((self.BOARD_ROWS, self.BOARD_COLS))
        self.current_player = self.X
        self.winner = None
        self.ended = False

    # Retornar uma lista de possíveis ações
    def get_possible_actions(self):
        actions = []
        for i in range(self.BOARD_ROWS):
            for j in range(self.BOARD_COLS):
                if self.board[i, j] == self.EMPTY:
                    actions.append((i, j))
        return actions

    # Realizar uma ação (marcar uma jogada)
    def step(self, action):

        i, j = action
        if self.board[i, j] == self.EMPTY:
            self.board[i, j] = self.current_player
            if self.is_winner(self.current_player):
                self.winner = self.current_player
                self.ended = True
            elif len(self.get_possible_actions()) == 0:
                self.ended = True
            else:
                self.current_player *= -1

    # Verificar se o jogador atual é o vencedor
    def is_winner(self, player):
        # Verificar linhas
        for i in range(self.BOARD_ROWS):
            if all(self.board[i, :] == player):
                return True
        # Verificar colunas
        for j in range(self.BOARD_COLS):
            if all(self.board[:, j] == player):
                return True
        # Verificar diagonais
        if all(self.board.diagonal() == player):
            return True
        if all(np.fliplr(self.board).diagonal() == player):
            return True
        return False

    # Retornar o estado atual do jogo (o tabuleiro)


    def get_state(self):

        return str(self.board.reshape(self.BOARD_ROWS * self.BOARD_COLS))

