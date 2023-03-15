import flappy_bird_gym
import numpy as np


from abstract.abstract_environment import AbstractEnvironment

#https://github.com/Talendar/flappy-bird-gym
class FlappyBirdEnv(AbstractEnvironment):

    def __init__(self):
        self.env = flappy_bird_gym.make("FlappyBird-v0")

    def reset(self):

        state = self.env.reset()

        state = np.reshape(state[0], [1, self.env.observation_space.shape[0]])

        return state
    def step(self, action):
        pass

    def render(self):
        pass

    def close(self):
        pass




