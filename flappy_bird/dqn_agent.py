from abstract.abstract_agent import AbstractAgent


class DQNAgent(AbstractAgent):

    def __init__(self):
        pass


    def train(self, env=None, num_episodes=1000, max_steps=500):
        state = env.reset()

        i=0

    def choose_action(self, state):
        pass

    def learn(self, state, action, reward, next_state, done):
        pass

    def save(self, filename):
        pass

    def load(self, filename):
        pass

