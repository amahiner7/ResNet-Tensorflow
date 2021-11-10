from tensorflow.keras.utils import Sequence
import numpy as np
from PIL import Image


class DataLoader(Sequence):
    def __init__(self, data, batch_size, data_shape=(200, 200, 3), shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = 0
        self.image_width = data_shape[0]
        self.image_height = data_shape[1]
        self.image_channel = data_shape[2]
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        indices_data = [self.data[index] for index in indices]
        images, labels = self.__data_generation(indices_data)

        return images, labels

    def on_epoch_end(self):
        self.indices = np.arange(len(self.data))

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, data):
        images = np.zeros((self.batch_size, self.image_width, self.image_height, self.image_channel))
        labels = np.zeros((self.batch_size, 1))

        for index, item in enumerate(data):  # i 는 파일 저장 경로가 포함된 파일 명
            image = Image.open(item.file_path).resize((self.image_width, self.image_height))
            image = (np.asarray(image) - 127.5) / 127.5
            images[index] = image
            labels[index] = item.age

        return images, labels
