from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, ZeroPadding2D, ReLU


class Conv1_Block(Layer):
    def __init__(self, name="Conv1_Block"):
        super().__init__(name=name)

        self.conv1_pad = ZeroPadding2D(padding=(3, 3))
        self.conv1 = Conv2D(filters=64,
                            kernel_size=(7, 7),
                            strides=(2, 2),
                            padding='valid',
                            use_bias=False,
                            kernel_initializer='he_normal')
        self.bn_conv1 = BatchNormalization()
        self.activation_1 = ReLU()
        self.pool1_pad = ZeroPadding2D(padding=(1, 1))

    def call(self, inputs):
        x = self.conv1_pad(inputs)
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = self.activation_1(x)
        x = self.pool1_pad(x)

        return x
