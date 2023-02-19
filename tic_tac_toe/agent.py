import random
import numpy as np


# Classe para o agente (Q-learning)
class TicTacToeAgent:
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    # Retornar o valor Q para um estado e ação específicos
    def get_q_value(self, state, action):
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0
        return self.q_table[(state, action)]

    # Atualizar o valor Q para um estado e ação específicos
    def update_q_value(self, state, action, value):
        old_value = self.get_q_value(state, action)
        self.q_table[(state, action)] = old_value + self.alpha * (value - old_value)

    # Escolher uma ação (com exploração / explotação)
    def choose_action(self, env):

        estrategy = np.random.uniform()

        possible_actions = env.get_possible_actions()

        if estrategy < self.epsilon:
            # Exploração: escolhe uma ação aleatória
            return random.choice(possible_actions)
        else:
            # Explotação: escolhe a melhor ação (com base nos valores Q)
            #q_values = [self.get_q_value(state, action) for action in possible_actions]
            max_q, q_values = self.calculate_q_value(env)

            if q_values.count(max_q) > 1:
                #seleciona todas as possíveis ações com valor de Q iguais ao máximo
                best_actions = []
                for i in range(len(possible_actions)):
                    if q_values[i] == max_q:
                        best_actions.append(i)
                # Se houver mais de uma ação com o mesmo valor máximo, escolhe aleatoriamente
                i = random.choice(best_actions)
            else:
                i = q_values.index(max_q)
            return possible_actions[i]

    # Treinar o agente usando Q-learning
    def train(self, env, num_episodes=1000):
        for episode in range(num_episodes):
            state = env.get_state()
            while not env.ended:

                action = self.choose_action(env)
                env.step(action)
                next_state = env.get_state()
                reward = 0

                if env.winner is not None:
                    reward = env.get_reward()

                max_q, q_values = self.calculate_q_value(env)
                new_q = reward + self.gamma * max_q
                self.update_q_value(state, action, new_q)

                state = next_state

    def calculate_q_value(self, env):

        state = env.get_state()
        p_actions = env.get_possible_actions()

        q_values = []
        for action in p_actions:
            q = self.get_q_value(state, action)
            q_values.append(q)

        if q_values:
            max_q = max(q_values)
        else:
            max_q = 0

        return max_q, q_values




