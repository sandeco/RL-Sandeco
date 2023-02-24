import gym
import numpy as np
import pickle

env = gym.make('CartPole-v1')

n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

# Definir os hiperparâmetros
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.9995
n_episodes = 5000
max_steps = 1000

# Define o número de intervalos para cada valor contínuo
num_bins = [10, 10, 10, 10]

# Define os limites de cada intervalo para cada valor contínuo
bin_limits = [(-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0), (-4.0, 4.0)]

# Define a forma da tabela Q-values
num_actions = 2
q_shape = tuple(num_bins + [num_actions])

# Cria uma tabela Q-values zerada
q_table = np.zeros(q_shape)


def discretize_state(state):
    # Discretiza cada valor contínuo usando intervalos fixos
    discretized_state = [np.digitize([s], np.linspace(l, u, n + 1))[0] for s, (l, u), n in
                         zip(state, bin_limits, num_bins)]

    # Verifica se os valores discretizados estão dentro dos limites
    discretized_state = [min(max(0, ds), nb - 1) for ds, nb in zip(discretized_state, num_bins)]

    return discretized_state


# Loop sobre o número de episódios
for episode in range(n_episodes):

    state = env.reset()[0]

    done = False
    t = 0

    total_reward = 0

    ran = 0
    tab = 0

    # Reduzir o valor de epsilon para cada episódio
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Loop sobre o número máximo de etapas por episódio
    while not done and t < max_steps:

        state_discrete = discretize_state(state)

        # Escolher uma ação usando epsilon-greedy
        if np.random.random() < epsilon:
            action = env.action_space.sample()
            ran +=1
        else:
            action = np.argmax(q_table[tuple(state_discrete)])
            tab +=1

        # Realizar a ação escolhida e receber a próxima observação, a recompensa e um indicador se o episódio terminou
        observation  = env.step(action)

        next_state = observation[0]
        reward = observation[1]

        # Atualizar a tabela Q

        next_state_discrete = discretize_state(next_state)


        # Calcular o valor Q antigo e o novo valor Q
        old_value = q_table[tuple(state_discrete) + (action,)]

        try:
            next_max = np.max(q_table[tuple(next_state_discrete) + (action,)])
        except:
            print("erro")

        new_value = old_value + learning_rate * (reward + discount_factor * next_max - old_value)

        q_table[tuple(state_discrete) + (action,)] += learning_rate * (reward + discount_factor * np.max(q_table[tuple(next_state_discrete) + (action,)]) - q_table[tuple(state_discrete) + (action,)])


        # Atualizar o estado atual
        state = next_state
        t += 1

        total_reward += reward


    # Imprimir a recompensa acumulada do episódio
    print("Episódio {}: Recompensa acumulada: {} Epsilon: {} random: {} q_table {}".format(episode, total_reward, epsilon, ran, tab))

# Fechar o ambiente
env.close()

fw = open('cartpolev1_agent', 'wb')
pickle.dump(q_table, fw)
fw.close()