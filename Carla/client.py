from dqn_agent import DQNAgent
from carla_env import CarlaEnv


#CarlaUE4.exe -quality-level=Low

from paths import Paths
from matplotlib import pyplot as plt

import glob
import os
import sys


egg_file = os.path.join(Paths.CARLA_EGG, "carla-*%d.%d-%s.egg")

try:
    sys.path.append(glob.glob(egg_file % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass



def main():

    env = None

    try:

        env = CarlaEnv()

        agent = DQNAgent(env=env,
                     lr=0.0001, gamma=0.99, epsilon_start=1.0,
                     epsilon_end=0.01, epsilon_decay=0.995)

        scores = agent.train(env, num_episodes=2000, max_steps=1000)

        plt.plot(scores)
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.show()

    except Exception as e:
        print(e)

    finally:
        if env is not None:
            env.close()

        env = None
        agent = None
        trainer = None

        print('FIM DO TREINAMENTO')

if __name__ == '__main__':
    main()
