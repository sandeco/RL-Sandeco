import itertools
import pickle

import gym
import numpy as np
import os


# Define a lista de valores para cada hiperparâmetro
learning_rates = [0.01, 0.1, 0.5]
discount_factors = [0.9, 0.95, 0.99]
epsilons = [1.0, 0.5, 0.1]
epsilon_mins = [0.01, 0.001]
epsilon_decays = [0.999, 0.9995, 0.9999]
num_episodes = [1000, 2000]
max_steps = [500, 1000]


# Realiza a busca em grade
best_reward = -np.inf
best_hyperparams = None

env = gym.make('CartPole-v1')


save_states = []

for hyperparams in itertools.product(learning_rates,
                                     discount_factors,
                                     epsilons,
                                     epsilon_mins,
                                     epsilon_decays,
                                     num_episodes,
                                     max_steps):


    save_states.append(hyperparams)


filename = 'index_params.pkl'
file_q_table = 'cartpolev1_agent'

# Define o número de intervalos para cada valor contínuo
num_bins = [10, 10, 10, 10]

num_actions = 2

# Verifica se já existe um arquivo com hiperparâmetros salvos
if os.path.exists(filename):
    with open(filename, 'rb') as f:
        best_hyperparams, best_reward, i = pickle.load(f)
        learning_rate, discount_factor, epsilon, epsilon_min, epsilon_decay, n_episodes, max_steps = best_hyperparams
        print("-----------------------------------")
        print("BEST REWARD : {}".format(best_reward))
        print("hyperparametrs -------------------")
        print("Learning rate : {}".format(learning_rate))
        print("Discount_factor : {}".format(discount_factor))
        print("Epsilon : {}".format(epsilon))
        print("Epsilon_min : {}".format(epsilon_min))
        print("epsilon_decay : {}".format(epsilon_decay))
        print("n_episodes : {}".format(n_episodes))
        print("max_steps : {}".format(max_steps))
        print("INDEX {}: ".format(i))

    with open(file_q_table, 'rb') as table:
        q_table = pickle.load(table)

    print('Loaded saved hyperparameters from', filename)
else:
    i = 0
    best_hyperparams = None
    best_reward = -np.inf
    episode_rewards = []

    # Define a forma da tabela Q-values
    q_shape = tuple(num_bins + [num_actions])

    # Cria uma tabela Q-values zerada
    q_table = np.zeros(q_shape)



while i < len(save_states):

    hyperparams = save_states[i]

    learning_rate, discount_factor, epsilon, epsilon_min, epsilon_decay, n_episodes, max_steps = hyperparams


    # Define os limites de cada intervalo para cada valor contínuo
    bin_limits = [(-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0), (-4.0, 4.0)]

    # Define a forma da tabela Q-values
    num_actions = 2
    q_shape = tuple(num_bins + [num_actions])

    # Cria uma tabela Q-values zerada
    #q_table = np.zeros(q_shape)

    def discretize_state(state):
        # Discretiza cada valor contínuo usando intervalos fixos
        discretized_state = [np.digitize([s], np.linspace(l, u, n + 1))[0] for s, (l, u), n in
                             zip(state, bin_limits, num_bins)]

        # Verifica se os valores discretizados estão dentro dos limites
        discretized_state = [min(max(0, ds), nb - 1) for ds, nb in zip(discretized_state, num_bins)]

        return discretized_state

    # Loop sobre o número de episódios
    episode_rewards = []

    for episode in range(n_episodes):
        state = env.reset()[0]
        done = False
        t = 0
        total_reward = 0

        # Reduzir o valor de epsilon para cada episódio
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Loop sobre o número máximo de etapas por episódio
        while not done and t < max_steps:

            state_discrete = discretize_state(state)

            # Escolher uma ação usando epsilon-greedy
            if np.random.random() < epsilon:
                action = env.action_space.sample()

            else:
                action = np.argmax(q_table[tuple(state_discrete)])


            # Realizar a ação escolhida e receber a próxima observação, a recompensa e um indicador se o episódio terminou
            observation = env.step(action)

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

            q_table[tuple(state_discrete) + (action,)] += learning_rate * (
                        reward + discount_factor * np.max(q_table[tuple(next_state_discrete) + (action,)]) - q_table[
                    tuple(state_discrete) + (action,)])

            # Atualizar o estado atual
            state = next_state
            t += 1
            total_reward += reward

            if t == max_steps and not done:
                done = True
                reward = -1

        print("Loop {}/{} Episódio {}/{} Best_reward {}".format(i,len(save_states), episode, n_episodes ,best_reward))

        episode_rewards.append(total_reward)

        # Verifica se a recompensa atual é maior que a melhor recompensa
        if total_reward > best_reward:

            best_reward = total_reward
            best_hyperparams = hyperparams

            print("\n-----------------------------------")
            print("BEST REWARD : {}".format(best_reward))
            print("hyperparametrs -------------------")
            print("Learning rate : {}".format(learning_rate))
            print("Discount_factor : {}".format(discount_factor))
            print("Epsilon : {}".format(epsilon))
            print("Epsilon_min : {}".format(epsilon_min))
            print("epsilon_decay : {}".format(epsilon_decay))
            print("n_episodes : {}".format(n_episodes))
            print("max_steps : {}".format(max_steps))
            print("-----------------------------------\n")

            fw = open('cartpolev1_agent', 'wb')
            pickle.dump(q_table, fw)
            fw.close()

    # Define a variável com os dados a serem salvos
    data_to_save = (best_hyperparams, best_reward, i )

    # Salva os dados em um arquivo
    with open(filename, 'wb') as f:
        pickle.dump(data_to_save, f)

    i = i +1


# Imprime os melhores hiperparâmetros encontrados e a melhor recompensa
print("Melhores hiperparâmetros:", best_hyperparams)
print("Melhor recompensa:", best_reward)