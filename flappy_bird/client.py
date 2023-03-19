
from flappy_bird_env import FlappyBirdEnv
from dqn_agent import DQNAgent


if __name__ == '__main__':

    env = FlappyBirdEnv(show=True)

    agente = DQNAgent(env=env,
                     lr=0.0001,
                     gamma=0.99,
                     epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.999)

    agente.train()
