import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCriticAgent:
    def __init__(self, state_size, action_size, lr_actor=0.0001, lr_critic=0.001, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)
        ).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

    def get_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.cpu().numpy()

    def update(self, rewards, log_probs, values):
        returns = []
        discounted_reward = 0
        for r in reversed(rewards):
            discounted_reward = r + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns).float().to(self.device)

        log_probs = torch.stack(log_probs)
        values = torch.stack(values).squeeze()

        advantages = returns - values
        actor_loss = -(advantages.detach() * log_probs).mean()
        critic_loss = advantages.pow(2).mean()

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def learn(self, env, num_episodes=1000, max_steps=1000):
        scores = []
        for i in range(num_episodes):
            state = env.reset()
            score = 0
            log_probs = []
            values = []
            for j in range(max_steps):
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                log_prob = torch.log(self.actor(torch.from_numpy(state).float().to(self.device))[action])
                value = self.critic(torch.from_numpy(state).float().to(self.device))
                log_probs.append(log_prob)
                values.append(value)
                state = next_state
                score += reward
                if done:
                    break
            self.update(rewards, log_probs, values)
            scores.append(score)
            print(f"Episode {i+1}/{num_episodes} - Score: {score}")
        return scores
