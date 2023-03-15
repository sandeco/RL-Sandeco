import cv2
from PIL import Image
import random
import numpy as np

from q_network import QNetwork


class DQNAgent:

    witdh  = 224
    height = 168

    def __init__(self, env, lr, gamma, epsilon_start, epsilon_end, epsilon_decay):

        self.n_actions = env.n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.input_shape = (self.witdh, self.height, 3)

        self.q_network = QNetwork(self.input_shape,
                                  self.n_actions,
                                  self.lr).create_model()

        self.target_q_network = QNetwork(self.input_shape,
                                         self.n_actions,
                                         self.lr).create_model()


        self.memory = ReplayMemory(10000)

        self.batch_size = 32
        self.batch_count = 0
        self.update_count = 0
        self.update_interval = 100

        self.action_space = [0, 1, 2]

    def get_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return np.random.choice(self.action_space)
        else:
            state = np.expand_dims(state, axis=0)

            pred = self.q_network.predict(state)
            return np.argmax(pred[0])

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        if self.batch_count < self.batch_size:
            self.batch_count +=1
            return

        self.batch_count = 0

        # amostragem aleatória de transições da memória
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)

        # cálculo do valor Q para cada ação usando o modelo target_q_network
        print("Buscando predição target")
        target_predict = self.target_q_network.predict(next_state_batch)

        print("Buscando predição Q")
        # cálculo do valor Q para cada ação usando o modelo q_network
        q = self.q_network.predict(state_batch)

        #Aplicando a função q-learning
        #Q(s, a) = Q(s, a) + α(r + γ * max(Q(s', a')) - Q(s, a))


        """
        A próxima linha representa todo esses comandos em python
        target_q = []
        for i in range(self.batch_size):
            if done_batch[i]:
                target_q.append(reward_batch[i])
            else:
                max_q = np.amax(target_predict[i])
                updated_q = reward_batch[i] + self.gamma * max_q
                target_q.append(updated_q)
        target_q = np.array(target_q)       
        """
        target_q = reward_batch + self.gamma * np.amax(target_predict, axis=1) * (1 - done_batch)



        # atualização do valor Q para a ação tomada na transição
        q[np.arange(self.batch_size), action_batch] = target_q

        print("treinamento do modelo q_network")
        self.q_network.fit(state_batch, q, epochs=1, verbose=1)

        # atualização do modelo target_q_network
        self.update_count += 1
        if self.update_count % self.update_interval == 0:
            self.target_q_network.set_weights(self.q_network.get_weights())

    def update_epsilon(self, episode):
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -1. * episode / self.epsilon_decay)
        return epsilon


    def preprocessing(self, img):
        imagem = Image.fromarray(img)
        imagem_redimensionada = imagem.resize((224, 168))
        img = np.array(imagem_redimensionada)
        cv2.imwrite('output/CarlaEnv-224x168.jpg', img)
        img = np.transpose(img, (1, 0, 2))

        print(img.shape)

        return img/255

    def train(self, env=None, num_episodes=2000, max_steps=1000):

        scores = []

        for i in range(num_episodes):
            state = self.preprocessing(env.reset())
            score = 0
            epsilon = self.update_epsilon(i)

            for j in range(max_steps):

                action = self.get_action(state, epsilon)

                next_state, reward, done, _ = env.step(action)
                next_state = self.preprocessing(next_state)

                self.memory.push(state, action, reward, next_state, done)

                self.learn()

                state = next_state
                score += reward

                if done:
                    break

            scores.append(score)

            print(f"Episode {i+1}/{num_episodes} - Score: {score} - Epsilon: {epsilon:.4f}")

        return scores


    def save_models(self):

        self.q_network.save("q_network.h5")
        self.target_q_network.save("target_q_network")



class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)

        state_batch = np.array([experience[0] for experience in batch])
        action_batch = np.array([experience[1] for experience in batch])
        reward_batch = np.array([experience[2] for experience in batch])
        next_state_batch = np.array([experience[3] for experience in batch])
        done_batch = np.array([experience[4] for experience in batch])

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.memory)





