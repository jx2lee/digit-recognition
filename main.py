#!/usr/bin/env python
import subprocess
import os


def get_info() -> None:
    print('[Detect VIN Num Using BlackBox Algorithm]')
    print('Usage:')
    print('******************************************')
    print('****** 1) CNN Tutorial                  **')
    print('****** 2) Blackbox Tutorial             **')
    print('****** 3) Train model                   **')
    print('****** 4) Test model                    **')
    print('******************************************')
    return


def run_cnn_tutorial() -> None:
    subprocess.call('cnn-example/cnn-example.py', shell=True)


def run_blackbox_tutorial() -> None:
    file_name = input('Enter the file name (ex.sample{1..3}.jpg): ')
    subprocess.run('python3 blackbox.py {}'.format(file_name), shell=True)


def run_train_model() -> None:
    from core.model import import_data, split_data, train_model

    # Variables
    epoch = int(input('Set Epoch(Number): '))
    batch_size = int(input('Set BatchSize(Number): '))
    model_checkpoint = os.getcwd() + '/' + 'res/model_checkpoint/LetterCNN.ckpt'
    train_data_path = os.getcwd() + '/tmp/Fnt'

    # Model Config
    x, y = import_data(train_data_path)
    x_train, x_test, y_train, y_test = split_data(x, y)

    # Train
    train_model(model_checkpoint, epoch, x_train, y_train, batch_size)


def run_test_model() -> None:
    from core.model import define_model, resize_input, prediction
    folder_name = input('Enter the folder name (ex.sample{1..3}): ')
    prediction(folder_name)


if __name__ == '__main__':
    get_info()
    number = int(input('Select Number: '))
    if number == 1:
        run_cnn_tutorial()
    elif number == 2:
        run_blackbox_tutorial()
    elif number == 3:
        run_train_model()
    elif number == 4:
        run_test_model()
    else:
        print('Enter the number 1 ~ 4')
        print('Good Bye..')