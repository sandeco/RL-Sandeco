
import tensorflow as tf



# Verifique se a GPU está sendo reconhecida pelo TensorFlow
print("Dispositivos físicos disponíveis: ", tf.config.list_physical_devices('GPU'))

# Verifique se o TensorFlow está usando a GPU
print("O TensorFlow está usando a GPU? ", tf.test.is_gpu_available())


