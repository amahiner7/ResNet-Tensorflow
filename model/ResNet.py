import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

from model.Conv1_Block import Conv1_Block
from model.ResidualBlock import ResidualBlock

from config.file_path import *
from config.hyper_parameters import *
from utils.LearningRateHistory import LearningRateHistory


class ResNet(Model):
    def __init__(self, input_shape, params, name="ResNet"):
        super().__init__(name=name)

        self.params = params

        self.model_input_shape = input_shape
        self.conv1_block = Conv1_Block()
        self.max_pooling2d_1 = MaxPooling2D((3, 3), 2)
        self.residual_blocks = []

        self.residual_blocks = [ResidualBlock(params=params[block_index], name="Residual_block_{}".format(block_index))
                                for block_index in range(len(params))]

        self.global_avg_pool2d = GlobalAveragePooling2D()

        self.feature_layer = Dense(units=256, activation='relu', kernel_initializer='he_normal', name='feature_layer')
        self.output_layer = Dense(units=1, activation='relu', kernel_initializer='he_normal', name='output')
        self.callbacks = []
        self.criterion = None
        self.optimizer = None

    def _check_compile(self):
        if self.criterion is None or self.optimizer is None:
            self.criterion = MeanSquaredError()
            self.optimizer = Adam()

        self.compile(optimizer=self.optimizer, loss=self.criterion)

    def make_callbacks(self, callbacks=None):
        if callbacks is not None:
            self.callbacks = callbacks
        else:
            model_check_point = ModelCheckpoint(filepath=MODEL_FILE_PATH,
                                                monitor='val_loss',
                                                save_weights_only=True,
                                                save_best_only=True,
                                                verbose=1)

            tensorboard = TensorBoard(log_dir=TENSORBOARD_LOG_DIR)
            learning_rate_history = LearningRateHistory(log_dir=TENSORBOARD_LEARNING_RATE_LOG_DIR)

            def _learning_rate_scheduler(epoch):
                if epoch+1 <= 10:
                    return LEARNING_RATE
                else:
                    return LEARNING_RATE * tf.math.exp(0.1 * (5 - epoch))

            learning_rate_scheduler = LearningRateScheduler(schedule=_learning_rate_scheduler, verbose=1)

            self.callbacks.append(model_check_point)
            self.callbacks.append(tensorboard)
            self.callbacks.append(learning_rate_scheduler)
            self.callbacks.append(learning_rate_history)

    def train_on_epoch(self, train_data, validation_data, epochs, verbose=1):
        self._check_compile()
        self.make_callbacks()

        history = self.fit(train_data,
                           validation_data=validation_data,
                           epochs=epochs,
                           callbacks=self.callbacks,
                           verbose=verbose)

        return history

    def call(self, inputs):
        x = self.conv1_block(inputs)
        x = self.max_pooling2d_1(x)

        for block_index in range(len(self.params)):
            x = self.residual_blocks[block_index](x)

        x = self.global_avg_pool2d(x)
        x = self.feature_layer(x)
        x = self.output_layer(x)

        return x

    def build_graph(self, input_shape, batch_size):
        input_layer = Input(shape=input_shape, batch_size=batch_size)

        return Model(inputs=input_layer, outputs=self.call(input_layer))

    def summary_model(self):
        temp_model = self.build_graph(input_shape=self.model_input_shape, batch_size=None)
        temp_model.summary()

    def load(self, model_file_name):
        model = self.build_graph(input_shape=self.model_input_shape, batch_size=None)
        model.load_weights(model_file_name)

        return model
