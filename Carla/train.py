class Train:
    def __init__(self, env, agent, num_episodes=2000, max_steps=1000):
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.max_steps = max_steps

    def run(self):

        scores = []

        for i in range(self.num_episodes):
            state = self.env.reset()
            score = 0
            epsilon = self.agent.update_epsilon(i)
            for j in range(self.max_steps):
                action = self.agent.get_action(state, epsilon)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.memory.push(state, action, reward, next_state, done)
                self.agent.learn()
                state = next_state
                score += reward
                if done:
                    break
            scores.append(score)
            print(f"Episode {i+1}/{self.num_episodes} - Score: {score} - Epsilon: {epsilon:.4f}")
        return scores
