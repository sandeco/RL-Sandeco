from env import Environment
from agent import Player

if __name__ == "__main__":
    # training

    #cria dois agentes
    p1 = Player("p1")
    p2 = Player("p2")

    #trina os agentes fazendo-os jogar entre si
    st = Environment(p1, p2)
    print("training...")
    st.train(1000)


    if p1.acc_reward >= p2.acc_reward:
        p1.savePolicy()
    else:
        p2.savePolicy()


    print("P1 - Total accumulated reward - {}".format(p1.acc_reward))
    print("P2 - Total accumulated reward - {}".format(p2.acc_reward))



