from mines_weeper import Minesweeper
from agent import Agent


def main():

    env = Minesweeper(size=3, num_mines=1)
    agent = Agent(num_states=env.size*env.size, num_actions=8)

    # Treinamento do agente
    num_episodes = 1000
    for episode in range(num_episodes):
        state = (0, 0)
        while not env.is_terminal(state):

            actions = env.get_actions(state)

            #valor único que representa o estado
            st = state[0] * env.size + state[1]

            #seleciona a ação com base no estado e possiveis ações
            action = agent.choose_action(state = st, possible_actions = actions)

            next_state = action

            reward = env.get_reward(next_state)
            agent.update_q_table(state[0]*env.size + state[1], actions.index(action), reward, next_state[0]*env.size + next_state[1])
            state = next_state

    # Jogo completo com o agente treinado
    state = (0, 0)
    while not env.is_terminal(state):
        actions = env.get_actions(state)
        action = agent.choose_action(state[0]*env.size + state[1], actions)
        next_state = action
        state = next_state
        env.show(state)

    print("Fim do jogo")


if __name__ == "__main__":
    main()