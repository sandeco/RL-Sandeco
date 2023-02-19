from agent import TicTacToeAgent
from env import TicTacToeEnvironment



if __name__ == '__main__':
    env = TicTacToeEnvironment()
    agent = TicTacToeAgent()

    print("Treinando...")
    agent.train(env)

    print("Jogando contra o agente treinado...")
    env.reset()
    while not env.ended:
        env.step(agent.choose_action(env.get_state(), env.get_possible_actions()))
        print(env.board)

    winner = env.winner

    if env.winner is None:
        print("Empate!")
    elif env.winner == env.X:
        print("Você ganhou!")
    else:
        print("O agente ganhou!")
