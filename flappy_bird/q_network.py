from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Flatten, Dropout


from tensorflow.keras.optimizers import Adam

class QNetwork:
    def __init__(self, input_shape, num_classes, lr):

        self.INPUT_SHAPE = input_shape
        self.CLASSES = num_classes
        self.lr = lr

    def create_model(self):

        conv_base = MobileNetV2(weights="imagenet", include_top=False, input_shape=self.INPUT_SHAPE)

        conv_base.trainable = False

        ## SINTONIA FINA NA MOBILENETV2

        """
        for i, layer in enumerate(conv_base.layers[:-5]):
           print(i, layer.name)
           layer.trainable = False
        """

        model = Sequential()
        model.add(conv_base)
        model.add(GlobalAveragePooling2D())
        model.add(BatchNormalization())

        #model.add(Flatten())
        model.add(Dense(128))

        # Camada de saída
        model.add(Dense(self.CLASSES, activation='linear'))

        # Compilação do modelo
        optimizer = Adam(learning_rate=self.lr)

        model.compile(optimizer=optimizer,
                      loss='mse')

        #print(model.summary())

        return model


