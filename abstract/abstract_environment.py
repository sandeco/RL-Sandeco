from abc import ABC, abstractmethod


class AbstractEnvironment(ABC):
    @abstractmethod
    def reset(self):
        """Reinicia o ambiente e retorna o estado inicial"""
        raise NotImplementedError

    @abstractmethod
    def step(self, action):
        """Executa a ação dada no ambiente e retorna o próximo estado,
        a recompensa e se o episódio terminou"""
        raise NotImplementedError

    @abstractmethod
    def n_actions(self):
        """Retorna a quantidade de ações do ambiente para determinado estado"""
        raise NotImplementedError


    @abstractmethod
    def get_actions(self):
        """Retorna as possíveis ações que podem ser executadas
        no ambiente para determinado estado"""
        raise NotImplementedError

    @abstractmethod
    def get_random_action(self):
        """Retorna, randomicamente, uma possível ação que pode ser executada
        no ambiente para determinado estado"""
        raise NotImplementedError



    @abstractmethod
    def render(self):
        """Renderiza o ambiente"""
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """Finaliza o ambiente"""
        raise NotImplementedError


