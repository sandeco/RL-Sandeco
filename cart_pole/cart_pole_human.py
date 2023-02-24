import gym
import numpy as np
import time
import pickle

env = gym.make('CartPole-v1', render_mode='human')

state = env.reset()[0]

fr = open('cartpolev1_agent', 'rb')
q_table = pickle.load(fr)
fr.close()

# Define o número de intervalos para cada valor contínuo
num_bins = [10, 10, 10, 10]

# Define os limites de cada intervalo para cada valor contínuo
bin_limits = [(-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0), (-4.0, 4.0)]


def discretize_state(state):
    # Discretiza cada valor contínuo usando intervalos fixos
    discretized_state = [np.digitize([s], np.linspace(l, u, n + 1))[0] for s, (l, u), n in
                         zip(state, bin_limits, num_bins)]

    # Verifica se os valores discretizados estão dentro dos limites
    discretized_state = [min(max(0, ds), nb - 1) for ds, nb in zip(discretized_state, num_bins)]

    return discretized_state

action = 0

while True:

    state = discretize_state(state)
    time.sleep(0.1)
    action = np.argmax(q_table[tuple(state)])
    # Realizar a ação escolhida e receber a próxima observação, a recompensa e um indicador se o episódio terminou
    observation = env.step(action)

    state = observation[0]

env.render()
env.step(0)
