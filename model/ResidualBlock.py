import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, ReLU


class ResidualBlock(Layer):
    def __init__(self, params, name="ResidualBlock"):
        super().__init__(name=name)

        self.filters_list = params['filters']
        self.kernel_size_list = params['kernel_size']
        self.strides_list = params['strides']
        self.padding_list = params['padding']
        self.num_blocks = params['num_blocks']

        self.block_list = []
        for index in range(self.num_blocks):
            block = self._create_block(index)
            self.block_list.append(block)

        self.shortcut_conv = Conv2D(filters=self.filters_list[2],
                                    kernel_size=self.kernel_size_list[0],
                                    strides=self.strides_list[0][0],
                                    padding=self.padding_list[0],
                                    use_bias=False,
                                    kernel_initializer='he_normal')

        self.shortcut_batch_norm = BatchNormalization()
        self.activation = ReLU()

    def _create_block(self, index):
        block = []

        if index == 0:
            block.append(Conv2D(filters=self.filters_list[0],
                                kernel_size=self.kernel_size_list[0],
                                strides=self.strides_list[0][0],
                                padding=self.padding_list[0],
                                use_bias=False,
                                kernel_initializer='he_normal'))
        else:
            block.append(Conv2D(filters=self.filters_list[0],
                                kernel_size=self.kernel_size_list[0],
                                strides=self.strides_list[0][1],
                                padding=self.padding_list[0],
                                use_bias=False,
                                kernel_initializer='he_normal'))

        block.append(BatchNormalization())
        block.append(ReLU())

        block.append(Conv2D(filters=self.filters_list[1],
                            kernel_size=self.kernel_size_list[1],
                            strides=self.strides_list[1],
                            padding=self.padding_list[1],
                            use_bias=False,
                            kernel_initializer='he_normal'))
        block.append(BatchNormalization())
        block.append(ReLU())

        block.append(Conv2D(filters=self.filters_list[2],
                            kernel_size=self.kernel_size_list[2],
                            strides=self.strides_list[2],
                            padding=self.padding_list[2],
                            use_bias=False,
                            kernel_initializer='he_normal'))
        block.append(BatchNormalization())

        return block

    def call(self, inputs):
        shortcut = inputs
        x = inputs

        for block_index in range(len(self.block_list)):
            block = self.block_list[block_index]

            for layer_index in range(len(block)):
                layer = block[layer_index]
                x = layer(x)

            if block_index == 0:
                shortcut = self.shortcut_conv(shortcut)
                shortcut = self.shortcut_batch_norm(shortcut)

            x = tf.add(x, shortcut)
            x = self.activation(x)

            shortcut = x

        return x
