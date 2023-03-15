import time
import flappy_bird_gym
from flappy_bird_gym.cli import random_agent_env

def random_agent_env():
    env = flappy_bird_gym.make("FlappyBird-v0")
    env.reset()
    score = 0
    while True:
        env.render()

        # Getting random action:
        action = env.action_space.sample()

        # Processing:
        obs, reward, done, _ = env.step(action)

        score += reward
        print(f"Obs: {obs}\n"
              f"Action: {action}\n"
              f"Score: {score}\n")

        time.sleep(1 / 30)

        if done:
            env.render()
            time.sleep(0.5)
            break

def main(mode):

    if mode == "human":
        flappy_bird_gym.original_game.main()
    elif mode == "random":
        random_agent_env()
    else:
        print("Invalid mode!")


if __name__ == '__main__':
    main("human")