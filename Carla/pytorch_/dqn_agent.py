import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from q_network import QNetwork

class DQNAgent:
    def __init__(self, env, lr, gamma, epsilon_start, epsilon_end, epsilon_decay):

        self.n_actions = env.n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = QNetwork(self.n_actions).to(self.device)
        self.target_q_network = QNetwork(self.n_actions).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        self.memory = ReplayMemory(10000)

        self.batch_size = 64
        self.update_count = 0
        self.update_interval = 100

        self.action_space = [0,1,2]

    def get_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return np.random.choice(self.action_space)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.q_network.eval()
            with torch.no_grad():
                action_values = self.q_network(state)
            self.q_network.train()
            return np.argmax(action_values.cpu().data.numpy())

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        batch_state = self.memory.sample(self.batch_size)

        print(batch_state.shape)

        with torch.no_grad():
            q_values = self.q_network(torch.from_numpy(batch_state).unsqueeze(0))
            action = q_values.argmax().item()

        """
        q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))
        next_q_values = self.target_q_network(next_state_batch).max(1)[0].detach()

        expected_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))
        loss = self.loss_fn(q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_count += 1
        if self.update_count % self.update_interval == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        """


    def update_epsilon(self, episode):
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * episode / self.epsilon_decay)
        return epsilon



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
        state_batch = np.array([experience[0] for experience in batch])/255
        action_batch = np.array([experience[1] for experience in batch])
        reward_batch = np.array([experience[2] for experience in batch])
        next_state_batch = np.array([experience[3] for experience in batch])/255
        done_batch = np.array([experience[4] for experience in batch])

        #return state_batch, action_batch, reward_batch, next_state_batch, done_batch

        return state_batch

    def __len__(self):
        return len(self.memory)

