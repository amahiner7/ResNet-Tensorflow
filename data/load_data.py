import csv
import random
import pandas as pd
from config.hyper_parameters import *
from config.file_path import *
from data.DataLoader import DataLoader
from data.UTKFaceData import UTKFaceData


def save_face_dataset(file_path, train_data, valid_data, test_data):
    file = open(file_path, 'w', encoding='utf-8', newline='')
    writer = csv.writer(file)
    writer.writerow(['age', 'gender', 'race', 'date', 'file_path', 'train_test_split'])
    for index, item in enumerate(train_data):
        writer.writerow(
            [item.age, item.gender, item.race, item.date, item.file_path, item.train_test_split])

    for index, item in enumerate(valid_data):
        writer.writerow(
            [item.age, item.gender, item.race, item.date, item.file_path, item.train_test_split])

    for index, item in enumerate(test_data):
        writer.writerow(
            [item.age, item.gender, item.race, item.date, item.file_path, item.train_test_split])
    file.close()


def load_face_dataset(file_path):
    data_frame = pd.read_csv(file_path)

    train_data = []
    valid_data = []
    test_data = []

    for row_index in data_frame.index:
        age = data_frame.iat[row_index, 0]
        gender = data_frame.iat[row_index, 1]
        race = data_frame.iat[row_index, 2]
        date = data_frame.iat[row_index, 3]
        file_path = data_frame.iat[row_index, 4]
        train_test_split = data_frame.iat[row_index, 5]

        utk_face_data = UTKFaceData(age=age, gender=gender, race=race, date=date,
                                    file_path=file_path, train_test_split=train_test_split)

        if train_test_split == 0:
            train_data.append(utk_face_data)
        elif train_test_split == 1:
            valid_data.append(utk_face_data)
        else:
            test_data.append(utk_face_data)

    return train_data, valid_data, test_data


def load_image_files(dataset_path, train_ratio=0.7, valid_ratio=0.2):
    image_file_list = os.listdir(dataset_path)

    data_list = []
    file_name_format_error_list = []

    for file in image_file_list:
        file_name, file_ext = os.path.splitext(file)
        file_info = file_name.split('_')

        if len(file_info) >= 4:
            data = UTKFaceData(age=file_info[0], gender=file_info[1], race=file_info[2],
                               date=file_info[3], file_path=os.path.join(dataset_path, file))
            data_list.append(data)
        else:
            file_name_format_error_list.append(file)

    random.shuffle(data_list)

    train_data_length = int(len(data_list) * train_ratio)
    valid_data_length = int(len(data_list) * valid_ratio)

    train_data = data_list[:train_data_length]  # Train: 70%
    valid_data = data_list[train_data_length:train_data_length + valid_data_length]  # Validation: 20%
    test_data = data_list[train_data_length + valid_data_length:len(data_list)]  # Test: 10%

    for item in train_data:
        item.train_test_split = 0
    for item in valid_data:
        item.train_test_split = 1
    for item in test_data:
        item.train_test_split = 2

    if len(file_name_format_error_list) > 0:
        print("=================================")
        print("File name format errors: ")
        for error_item in file_name_format_error_list:
            print(error_item)
        print("=================================")

    return train_data, valid_data, test_data


def get_data_sample(data, sample_ratio):
    data_len = len(data)
    sample_count = int(data_len * sample_ratio)

    return data[:sample_count]


def load_data(sample_ratio=1.0):
    face_dataset_file_path = os.path.join(UTKFace_DIR, FACE_DATASET_FILE_NAME)
    if os.path.isfile(face_dataset_file_path):
        train_data, valid_data, test_data = load_face_dataset(file_path=face_dataset_file_path)
        print("Load dataset file: {}.".format(face_dataset_file_path))
    else:
        train_data, valid_data, test_data = load_image_files(dataset_path=IMAGE_DATA_FILE_PATH)
        save_face_dataset(file_path=face_dataset_file_path,
                          train_data=train_data,
                          valid_data=valid_data,
                          test_data=test_data)
        print("Save dataset file: {}.".format(face_dataset_file_path))

    if sample_ratio < 1.0:
        train_data = get_data_sample(train_data, sample_ratio=sample_ratio)
        valid_data = get_data_sample(valid_data, sample_ratio=sample_ratio)
        test_data = get_data_sample(test_data, sample_ratio=sample_ratio)

    print("Total data length: ", len(train_data) + len(valid_data) + len(test_data))
    print("train_data length: ", len(train_data))
    print("valid_data length: ", len(valid_data))
    print("test_data length: ", len(test_data))

    train_data_loader = DataLoader(data=train_data, batch_size=BATCH_SIZE,
                                   data_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL), shuffle=True)

    valid_data_loader = DataLoader(data=valid_data, batch_size=BATCH_SIZE,
                                   data_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL), shuffle=True)

    test_data_loader = DataLoader(data=test_data, batch_size=BATCH_SIZE,
                                  data_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL), shuffle=True)

    return train_data_loader, valid_data_loader, test_data_loader
