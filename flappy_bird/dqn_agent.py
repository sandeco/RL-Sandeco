from abstract.abstract_agent import AbstractAgent

from q_network import QNetwork

import cv2
import time
import numpy as np

from abstract.replay_memory import ReplayMemory

class DQNAgent(AbstractAgent):

    def __init__(self, env, lr, gamma, epsilon_start, epsilon_end, epsilon_decay):

        self.env = env
        self.lr = lr
        self.gamma = gamma

        self.epsilon = 1
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.score = 0

        self.replay_memory = ReplayMemory(10000)
        self.batch_size = 64
        self.batch_count = 0

        self.width = 224
        self.height = 224

        self.INPUT_SHAPE = (self.width, self.height, 3)
        self.actions = self.env.n_actions()

        self.q_network = QNetwork(self.INPUT_SHAPE, self.actions, lr).create_model()
        self.target_q_network = QNetwork(self.INPUT_SHAPE, self.actions, lr).create_model()

        self.update_count = 0
        self.update_interval = 100

        self.random_action = 0
        self.net_action = 0

        self.action_probabilities = np.array([0.5, 0.5])
        self.action_probabilities[0] *= 10
        self.action_probabilities /= np.sum(self.action_probabilities)

        self.penalties = 0

        self.training = False

    def preprocessing(self, img):
        cv2.imwrite('output/flappy-288-512.jpg', np.transpose(img, (1, 0, 2)))

        # Redimensionar a imagem usando o método cv2.resize
        img_resized = cv2.resize(img, (self.height, self.width), interpolation=cv2.INTER_AREA)
        cv2.imwrite('output/flappy-224-224.jpg', np.transpose(img_resized, (1, 0, 2)))

        # Normalizar a imagem
        img_normalized = img_resized / 255.0

        return img_normalized

    def train(self, num_episodes=1000, max_steps=500):

        scores = [100]

        for i in range(num_episodes):

            state = self.env.reset()

            state = state[0]

            state_pro = self.preprocessing(state)
            self.update_epsilon(i)

            score = 0

            for j in range(max_steps):

                #self.env.show_state(state)

                action = self.choose_action(state_pro, self.epsilon)

                next_state, reward, done, _, info = self.env.step(action)

                if reward == 1:
                    reward = 100
                    self.training = True
                    self.penalties = self.penalties - 10


                if done:
                    self.penalties +=1
                    reward = reward - self.penalties/100

                next_state_pro = self.preprocessing(next_state)

                self.replay_memory.push(state_pro, action, reward, next_state_pro, done)
                score += reward

                if self.training:
                    self.learn()

                state = next_state
                state_pro = state_pro

                if done:
                    break


            scores.append(score)

            print(f"Episode {i+1}/{num_episodes} - Score: {score} - Epsilon: {self.epsilon:.4f}")

        return scores


    def choose_action(self, state, epsilon):

        if np.random.rand() <= epsilon:
            action = np.random.choice(self.env.get_actions(), p=self.action_probabilities)
        else:
            state = np.expand_dims(state, axis=0)
            pred = self.q_network.predict(state)
            action = np.argmax(pred[0])


        #print("RANDON {}, NETWORK {}".format(self.random_action, self.net_action))

        return action


    def is_batch(self):

        if self.batch_count >= self.batch_size:
            self.batch_count = 0
            return True

        self.batch_count+=1
        return False

    def learn(self):

        if self.is_batch():
            # amostragem aleatória de transições da memória
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_memory.sample(self.batch_size)

            # cálculo do valor Q para cada ação usando o modelo target_q_network
            target_predict = self.target_q_network.predict(next_state_batch)


            # cálculo do valor Q para cada ação usando o modelo q_network
            q = self.q_network.predict(state_batch)

            target_q = reward_batch + self.gamma * np.amax(target_predict, axis=1) * (1 - done_batch)

            # atualização do valor Q para a ação tomada na transição
            q[np.arange(self.batch_size), action_batch] = target_q


            self.q_network.fit(state_batch, q, epochs=1, verbose=1)

            # atualização do modelo target_q_network
            self.update_count += 1
            if self.update_count % self.update_interval == 0:
                self.target_q_network.set_weights(self.q_network.get_weights())


    def update_epsilon(self, episode):

        self.epsilon = self.epsilon * self.epsilon_decay

        #epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
        #    -1. * episode / self.epsilon_decay)

    def save(self):
        pass

    def load(self):
        pass



