from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import Dropout, Dense, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint

from data_preparation import load_data_joint
from feature_generator_joint import generator

import numpy as np


def jan_joint(filter_density, dropout, input_shape, batchNorm=False, dense_activation='relu', channel=1):
    if channel == 1:
        reshape_dim = (1, input_shape[0], input_shape[1])
        channel_order = 'channels_first'
    else:
        reshape_dim = input_shape
        channel_order = 'channels_last'

    input = Input(shape=reshape_dim)

    if batchNorm:
        x = BatchNormalization(axis=1, input_shape=reshape_dim)(input)
    else:
        x = input

    x = Conv2D(int(10 * filter_density), (3, 7), padding="valid",
                       data_format=channel_order, activation='relu')(x)

    x = MaxPooling2D(pool_size=(3, 1), padding='valid', data_format=channel_order)(x)

    x = Conv2D(int(20 * filter_density), (3, 3), padding="valid",
                       data_format=channel_order, activation='relu')(x)

    x = MaxPooling2D(pool_size=(3, 1), padding='valid', data_format=channel_order)(x)

    if dropout:
        x = Dropout(dropout)(x)  # test Schluter dataset, comment in jingju dataset

    x = Flatten()(x)

    x = Dense(units=256, activation=dense_activation)(x)

    if dropout:
        x = Dropout(dropout)(x)

    syllable_out = Dense(1, activation='sigmoid', name='syllable_output')(x)
    phoneme_out = Dense(1, activation='sigmoid', name='phoneme_output')(x)

    model = Model(inputs=input, outputs=[syllable_out, phoneme_out])

    optimizer = Adam()

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer)

    print(model.summary())

    return model


# this is not used in the paper
def jan_joint_deep(filter_density, dropout, input_shape, batchNorm=False, dense_activation='relu', channel=1):
    "less deep architecture"
    if channel == 1:
        reshape_dim = (1, input_shape[0], input_shape[1])
        channel_order = 'channels_first'
    else:
        reshape_dim = input_shape
        channel_order = 'channels_last'

    padding = "same"

    input = Input(shape=reshape_dim)

    if batchNorm:
        x = BatchNormalization(axis=1, input_shape=reshape_dim)(input)
    else:
        x = input

    x = Conv2D(int(10 * filter_density), (3, 7), padding="valid",
                       input_shape=reshape_dim,
                       data_format=channel_order, activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 1), padding='valid', data_format=channel_order)(x)

    x = Conv2D(int(20 * filter_density), (3, 3), padding="valid",
                       data_format=channel_order, activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 1), padding='valid', data_format=channel_order)(x)

    if dropout:
        x = Dropout(dropout)(x)

    # replacement of the dense layer
    x = Conv2D(int(60 * filter_density), (3, 3), padding=padding,
                       data_format=channel_order, activation='relu')(x)
    x = BatchNormalization(axis=1)(x)

    x = Conv2D(int(60 * filter_density), (3, 3), padding=padding,
                       data_format=channel_order, activation='relu')(x)
    x = BatchNormalization(axis=1)(x)

    x = Conv2D(int(60 * filter_density), (3, 3), padding=padding,
                       data_format=channel_order, activation='relu')(x)
    x = BatchNormalization(axis=1)(x)

    x = Flatten()(x)

    if dropout:
        x = Dropout(dropout)(x)

    syllable_out = Dense(1, activation='sigmoid', name='syllable_output')(x)
    phoneme_out = Dense(1, activation='sigmoid', name='phoneme_output')(x)

    model = Model(inputs=input, outputs=[syllable_out, phoneme_out])

    optimizer = Adam()

    model.compile(loss='binary_crossentropy',
                  loss_weights=[0.8375, 0.1625],
                  optimizer=optimizer)

    return model


def model_train_validation(model_0,
                           batch_size,
                           patience,
                           input_shape,
                           path_feature_data,
                           idx_train,
                           Y_syllable_train,
                           Y_phoneme_train,
                           sample_weights_syllable_train,
                           sample_weights_phoneme_train,
                           idx_val,
                           Y_syllable_val,
                           Y_phoneme_val,
                           sample_weights_syllable_val,
                           sample_weights_phoneme_val,
                           file_path_model,
                           filename_log,
                           channel):

    """
    train the model with validation early stopping and retrain the model with whole training dataset
    :param model_0:
    :param batch_size:
    :param patience:
    :param input_shape:
    :param path_feature_data:
    :param indices_train:
    :param Y_train:
    :param sample_weights_train:
    :param indices_validation:
    :param Y_validation:
    :param sample_weights_validation:
    :param indices_all:
    :param Y_train_validation:
    :param sample_weights:
    :param class_weights:
    :param file_path_model:
    :param filename_log:
    :return:
    """

    callbacks = [ModelCheckpoint(file_path_model, monitor='val_syllable_output_loss', verbose=0, save_best_only=True),
                EarlyStopping(monitor='val_syllable_output_loss', patience=patience, verbose=0),
                 CSVLogger(filename=filename_log, separator=';')]

    print("start training with validation...")

    steps_per_epoch_train = int(np.ceil(len(idx_train) / batch_size))
    steps_per_epoch_val = int(np.ceil(len(idx_val) / batch_size))

    generator_train = generator(path_feature_data=path_feature_data,
                                indices=idx_train,
                                number_of_batches=steps_per_epoch_train,
                                file_size=batch_size,
                                input_shape=input_shape,
                                labels_syllable=Y_syllable_train,
                                labels_phoneme=Y_phoneme_train,
                                sample_weights_syllable=sample_weights_syllable_train,
                                sample_weights_phoneme=sample_weights_phoneme_train,
                                channel=channel)

    generator_val = generator(path_feature_data=path_feature_data,
                              indices=idx_val,
                              number_of_batches=steps_per_epoch_val,
                              file_size=batch_size,
                              input_shape=input_shape,
                              labels_syllable=Y_syllable_val,
                              labels_phoneme=Y_phoneme_val,
                              sample_weights_syllable=sample_weights_syllable_val,
                              sample_weights_phoneme=sample_weights_phoneme_val,
                              channel=channel)

    model_0.fit_generator(generator=generator_train,
                          steps_per_epoch=steps_per_epoch_train,
                          epochs=500,
                          validation_data=generator_val,
                          validation_steps=steps_per_epoch_val,
                          callbacks=callbacks,
                          verbose=2)


def train_model_validation(filename_train_validation_set,
                           filename_labels_syllable_train_validation_set,
                           filename_labels_phoneme_train_validation_set,
                           filename_sample_weights_syllable,
                           filename_sample_weights_phoneme,
                           filter_density,
                           dropout,
                           input_shape,
                           file_path_model,
                           filename_log,
                           channel=1):
    """
    train model with validation
    """

    idx_train, Y_syllable_train, Y_phoneme_train, sample_weights_syllable_train, sample_weights_phoneme_train, \
    idx_val, Y_syllable_val, Y_phoneme_val, sample_weights_syllable_val, sample_weights_phoneme_val, \
    indices_all, Y_syllable_train_validation, Y_phoneme_train_validation = \
    load_data_joint(filename_labels_syllable_train_validation_set,
                   filename_labels_phoneme_train_validation_set,
                   filename_sample_weights_syllable,
                   filename_sample_weights_phoneme)

    # this is the model used in the paper
    model_0 = jan_joint(filter_density,
                        dropout,
                        input_shape,
                        batchNorm=False,
                        dense_activation='sigmoid',
                        channel=channel)

    # model_0 = jan_joint_deep(filter_density,
    #                          dropout,
    #                          input_shape,
    #                          batchNorm=False,
    #                          dense_activation='sigmoid',
    #                          channel=channel)

    batch_size = 256
    patience = 15

    model_train_validation(model_0,
                           batch_size,
                           patience,
                           input_shape,
                           filename_train_validation_set,
                           idx_train,
                           Y_syllable_train,
                           Y_phoneme_train,
                           sample_weights_syllable_train,
                           sample_weights_phoneme_train,
                           idx_val,
                           Y_syllable_val,
                           Y_phoneme_val,
                           sample_weights_syllable_val,
                           sample_weights_phoneme_val,
                           file_path_model,
                           filename_log,
                           channel)

if __name__ == '__main__':
    model = jan_joint(1,
                      0.5,
                      (80, 15),
                      batchNorm=False,
                      dense_activation='sigmoid',
                      channel=1)