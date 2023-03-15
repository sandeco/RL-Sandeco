import gym
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.layers import LeakyReLU

from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.models import load_model

from collections import deque

import pickle
import os

# Cria o ambiente CartPole-v1
env = gym.make('CartPole-v1')

# Define os hiperparâmetros
gamma = 0.95  # Fator de desconto
epsilon = 1.0  # Taxa de exploração
epsilon_decay = 0.995  # Decaimento da taxa de exploração
epsilon_min = 0.01  # Valor mínimo da taxa de exploração
learning_rate = 0.001  # Taxa de aprendizado
batch_size = 128  # Tamanho do lote
memory = deque(maxlen=2000)  # Buffer de replay

model_file = "model.h5"
hy_params_file = "hyperparams.pkl"

# Carrega se existir o modelo do cart-pole

if os.path.isfile(model_file):

    print("CARREGANDO MODELO")

    model = load_model(model_file)

    with open(hy_params_file, 'rb') as f:
        epsilon, start_episode, end_episode, best_score = pickle.load(f)

else:
    print("CRIANDO MODELO")
    # Define a rede neural para estimar a função Q
    model = Sequential()

    model.add(Dense(1024 , input_shape=env.observation_space.shape))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))

    model.add(Dense(env.action_space.n, activation='linear'))

    optimizer = adam_v2.Adam(learning_rate=learning_rate)
    model.compile(loss='mse', optimizer=optimizer)

    start_episode = 1
    best_score = 0

end_episode = 3001

print("\n\n*************BEST SCORE ENCONTRADO*************\n\n")
print(f"Episódio {start_episode}: Pontuação = {best_score}, Taxa de Exploração = {epsilon}")


# Define a função de seleção de ação usando a política epsilon-greedy
def select_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(model.predict(state)[0])


# Loop principal do treinamento

for episode in range(start_episode, end_episode):
    state = env.reset()

    test = 1, env.observation_space.shape[0]

    state = np.reshape(state[0], [1, env.observation_space.shape[0]])
    done = False
    score = 0
    while not done:
        action = select_action(state, epsilon)

        step_params = env.step(action)

        next_state, reward, done, _, __ = step_params
        next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
        score += reward
        memory.append((state, action, reward, next_state, done))
        state = next_state
        if len(memory) > batch_size:
            minibatch = np.array(list(memory))[np.random.choice(len(memory), batch_size, replace=False), :]
            X = np.vstack(minibatch[:, 0])
                        
            y = model.predict(X)
            next_states = np.vstack(minibatch[:, 3])
            next_Qs = np.amax(model.predict(next_states), axis=1)
            targets = minibatch[:, 2] + gamma * next_Qs * (1 - minibatch[:, 4])
            actions = minibatch[:, 1].astype(int)
            y[np.arange(batch_size), actions] = targets
            model.fit(X, y, epochs=3, verbose=0)

        if done:
            epsilon = max(epsilon * epsilon_decay, epsilon_min)

            # SAVE PARAMS
            data_to_save = (epsilon, episode, end_episode, best_score)

            # Salva os dados em um arquivo
            with open(hy_params_file, 'wb') as f:
                pickle.dump(data_to_save, f)
                print("\n\n salvando pickes EPISODE {}".format(episode))

        if score > best_score:
            best_score = score

            print("\n\n*************BEST SCORE ENCONTRADO*************\n\n")
            print(f"Episódio {episode}: Pontuação = {score}, Taxa de Exploração = {epsilon}")

            # SAVE MODEL
            model.save(model_file)

