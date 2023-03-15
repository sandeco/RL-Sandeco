from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, LeakyReLU, Flatten, BatchNormalization

from tensorflow.keras.optimizers import Adam

class QNetwork:
    def __init__(self, input_shape, num_classes, lr):

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lr = lr

    def create_model(self):
        # Criação do modelo
        model = Sequential()

        # Camadas convolucionais
        model.add(Conv2D(32, (3, 3),  input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))

        model.add(Conv2D(32, (3, 3)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Camada densa
        model.add(GlobalAveragePooling2D())

        # Camada de saída
        model.add(Dense(self.num_classes, activation='softmax'))

        # Compilação do modelo
        optimizer = Adam(learning_rate=self.lr)

        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        print(model.summary())

        return model


