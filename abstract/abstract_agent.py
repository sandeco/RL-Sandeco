from abc import ABC, abstractmethod

class AbstractAgent(ABC):



    @abstractmethod
    def train(self, env=None, num_episodes=1000, max_steps=500):
        """Treina o agente"""
        pass


    @abstractmethod
    def choose_action(self, state):
        """Seleciona uma ação para o estado atual"""
        pass


    @abstractmethod
    def learn(self):
        """Atualiza a política do agente com base na experiência de um episódio"""
        pass

    @abstractmethod
    def save(self):
        """Salva o modelo do agente em um arquivo"""
        pass

    @abstractmethod
    def load(self):
        """Carrega o modelo do agente de um arquivo"""
        pass
