from config.model_parameters import *
from model.ResNet import ResNet
from data.load_data import *
from utils.common import *

make_directory()
train_data_loader, valid_data_loader, _ = load_data()

model = ResNet(input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL), params=RESNET50_MODEL_PARAMS)
model.summary_model()

if __name__ == '__main__':
    history = model.train_on_epoch(train_data=train_data_loader,
                                   validation_data=valid_data_loader,
                                   epochs=NUM_EPOCHS)

    display_loss(history.history)
