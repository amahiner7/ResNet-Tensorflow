import os
import glob
from PIL import Image

from model.ResNet import ResNet
from config.model_parameters import *
from config.hyper_parameters import *
from data.load_data import load_data
from utils.common import *


LOAD_MODEL_FILE_DIR = "./model_files/20211110-003436_kaiming_he_lambda_lr_200x200"


def load_model():
    save_model_file_names = sorted(glob.glob(os.path.join(LOAD_MODEL_FILE_DIR, "*.h5")))

    if len(save_model_file_names) > 0:
        model = ResNet(input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL), params=RESNET50_MODEL_PARAMS)

        last_file = save_model_file_names[len(save_model_file_names) - 1]
        model = model.load(last_file)
        model.summary()

        print("{} is loaded.".format(last_file))

        return model
    else:
        raise Exception("It can't find model files.")


_, _, test_data_loader = load_data()

model = load_model()

num_rows = 5
num_cols = 3
num_samples = num_rows * num_cols
sample_dataset = test_data_loader.data[:num_samples]

plt.figure()
for index, data in enumerate(sample_dataset):
    label = data.age

    plt.subplot(num_rows, num_cols, index + 1)

    origin_image = Image.open(data.file_path).resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    image = np.asarray(origin_image) / 255.0
    image = image.reshape(1, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL)

    y_predict = model.predict(image)
    y_predict = (y_predict.squeeze())

    plot_image(pred=y_predict, label=label, image=origin_image)

plt.tight_layout()
plt.show()
