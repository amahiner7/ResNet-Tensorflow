import numpy as np
import matplotlib.pyplot as plt


def display_loss(history):
    train_loss = history['loss']
    val_loss = history['val_loss']

    # 그래프로 표현
    x_len = np.arange(len(train_loss))
    plt.figure()
    plt.plot(x_len, val_loss, marker='.', c="blue", label='Validation loss')
    plt.plot(x_len, train_loss, marker='.', c="red", label='Train loss')
    # 그래프에 그리드를 주고 레이블을 표시
    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.show()

    if history.get('learning_rate') is not None:
        learning_rate = history['learning_rate']
        x_len = np.arange(len(learning_rate))
        plt.clf()
        plt.figure()
        plt.plot(x_len, learning_rate, marker='.', c="yellow", label='Learning rate')
        plt.legend(loc='upper right')
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('Learning rate')
        plt.title('Learning rate')
        plt.show()


def plot_image(pred, label, image):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(image)

    if np.math.fabs(pred - float(label)) < 10.0:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("Pred: {:.1f} | Label: {}".format(pred, label), color=color)


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    this_plot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    this_plot[predicted_label].set_color('red')
    this_plot[true_label].set_color('blue')
