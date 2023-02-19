import random
import numpy as np

class TicTacToeAgent:
    def __init__(self, alpha=0.6, gamma=0.9, epsilon=0.1):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.pause_factor = 1000
        self.pause_cont = 0
        self.pause_next = 0
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
    def choose_action(self, state, possible_actions):
        if np.random.uniform() < self.epsilon:
            # Exploração: escolhe uma ação aleatória
            return random.choice(possible_actions)
        else:
            # Explotação: escolhe a melhor ação (com base nos valores Q)
            q_values = [self.get_q_value(state, action) for action in possible_actions]
            max_q = max(q_values)
            if q_values.count(max_q) > 1:
                # Se houver mais de uma ação com o mesmo valor máximo, escolhe aleatoriamente
                best_actions = [i for i in range(len(possible_actions)) if q_values[i] == max_q]
                i = random.choice(best_actions)
            else:
                i = q_values.index(max_q)
            return possible_actions[i]


    # Treinar o agente usando Q-learning
    def train(self, env, num_episodes=1000):

        for episode in range(num_episodes):

            if env.ended:
                env.reset()

            state = env.get_state()
            while not env.ended:
                possible_actions = env.get_possible_actions()
                action = self.choose_action(state, possible_actions)
                env.step(action)
                next_state = env.get_state()
                reward = 0
                if env.winner is not None:
                    reward = 1 if env.winner == env.X else -1

                q_values = [self.get_q_value(next_state, a) for a in env.get_possible_actions()]
                if q_values:
                    max_q = max(q_values)
                else:
                    max_q = 0

                self.update_q_value(state, action, reward + self.gamma * max_q)
                state = next_state

        print("FIM DO TREINAMENTO")
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

