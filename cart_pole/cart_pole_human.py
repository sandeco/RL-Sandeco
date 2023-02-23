import gym
import numpy as np
import time

env = gym.make('CartPole-v1', render_mode='human')

(state,_) = env.reset()

x = state[0]
x_dot = state[1]
theta = state[2]
theta_dot = state[3]

obs = env.observation_space

while True:
    time.sleep(0.2)
    env.render()
    env.step(0)
