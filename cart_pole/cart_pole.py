import gym
import numpy as np

env = gym.make('CartPole-v1')

n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

# Definir os hiperparâmetros
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
n_episodes = 1000
max_steps = 500

Q = np.zeros((n_states, n_actions))


# Definir os limites para cada componente do estado
pos_bins = np.linspace(-4.8, 4.8, 9)
vel_bins = np.linspace(-2.0, 2.0, 9)
ang_bins = np.linspace(-0.5, 0.5, 9)
ang_vel_bins = np.linspace(-3.5, 3.5, 9)

def discretize_state(state):

    pos_bin = np.digitize(state[0][0], pos_bins)
    vel_bin = np.digitize(state[0][1], vel_bins)
    ang_bin = np.digitize(state[0][2], ang_bins)
    ang_vel_bin = np.digitize(state[0][3], ang_vel_bins)
    return pos_bin, vel_bin, ang_bin, ang_vel_bin


# Loop sobre o número de episódios
for episode in range(n_episodes):

    state = env.reset()

    done = False
    t = 0

    # Reduzir o valor de epsilon para cada episódio
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Loop sobre o número máximo de etapas por episódio
    while not done and t < max_steps:
        # Escolher uma ação usando epsilon-greedy
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        # Realizar a ação escolhida e receber a próxima observação, a recompensa e um indicador se o episódio terminou
        observation  = env.step(action)

        next_state = observation[0]
        reward = observation[1]

        # Atualizar a tabela Q
        state_discrete = discretize_state(state)
        next_state_discrete = discretize_state(next_state)
        Q[state_discrete][action] += learning_rate * (reward + discount_factor * np.max(Q[next_state_discrete]) - Q[state_discrete][action])


        # Atualizar o estado atual
        state = next_state
        t += 1

    # Imprimir a recompensa acumulada do episódio
    print("Episódio {}: Recompensa acumulada = {}".format(episode, t))

# Fechar o ambiente
env.close()