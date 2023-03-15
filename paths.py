import os

class Paths():
    """
    Classe responsável por armazenar e fornecer caminhos
    de arquivos e pastas utilizados no sistema.
    """
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    CARLA_EGG = os.path.join(ROOT_DIR,'carla-sim','PythonAPI','carla','dist')
