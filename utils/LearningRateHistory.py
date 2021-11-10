import tensorflow as tf


class LearningRateHistory(tf.keras.callbacks.Callback):
    def __init__(self, log_dir=None):
        self.lr_history_list = []
        self.summary_writer = None

        if log_dir is not None:
            self.summary_writer = tf.summary.create_file_writer(logdir=log_dir)
            self.summary_writer.set_as_default()

    def on_epoch_begin(self, epoch, logs):
        current_learning_rate = float(tf.keras.backend.get_value(self.model.optimizer._decayed_lr(tf.float32)))
        self.lr_history_list.append(current_learning_rate)

        if self.summary_writer is not None:
            tf.summary.scalar('Learning rate', data=current_learning_rate, step=epoch)
