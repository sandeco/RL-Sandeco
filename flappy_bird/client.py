
from flappy_bird_env import FlappyBirdEnv
from dqn_agent import DQNAgent


if __name__ == '__main__':

    env = FlappyBirdEnv()
    agente = DQNAgent()

    agente.train(env)
