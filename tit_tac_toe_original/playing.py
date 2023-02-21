from env import Environment
from agent import Player
from human_player import HumanPlayer

if __name__ == "__main__":
    # training

    #cria dois agentes
    p1 = Player("p1")
    p2 = HumanPlayer("Humano")


    p1.loadPolicy("policy_agent")

    env = Environment(p1,p2)

    env.play()
