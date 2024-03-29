import cv2
import numpy as np
import flappy_bird_gymnasium
import gymnasium

from abstract.abstract_environment import AbstractEnvironment

# https://github.com/markub3327/flappy-bird-gymnasium
class FlappyBirdEnv(AbstractEnvironment):

    def __init__(self, show=False):
        self.env = gymnasium.make("FlappyBird-rgb-v0")
        self.SHOW = show

        self.actions = np.array((0,1))

    def reset(self):

        state = self.env.reset()
        return state

    def n_actions(self):
        actions = len(self.get_actions())
        return actions

    def get_actions(self):
        actions = [0,1]
        return actions

    def get_random_action(self):
        action = self.env.action_space.sample()
        return action

    def show_state(self, state):
        if self.SHOW:
            cv2.imshow("", np.transpose(state, (1, 0, 2)))
            cv2.waitKey(1)

    def step(self, action):
        obs, reward, done, _, info = self.env.step(action)
        return obs, reward, done, _, info

    def render(self):
        self.env.render()

    def close(self):
        pass




