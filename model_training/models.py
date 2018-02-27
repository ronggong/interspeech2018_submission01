"""multi-filters HSMM emission probability model"""
import numpy as np
from keras import backend as K
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D
from keras.layers import Dropout, Dense, Flatten, BatchNormalization
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.optimizers import Adam

from feature_generator import generator
from data_preparation import load_data


def jan_original(filter_density, dropout, input_shape, batchNorm=False, dense_activation='relu', channel=1):
    """this model is not used in the paper"""
    if channel == 1:
        reshape_dim = (1, input_shape[0], input_shape[1])
        channel_order = 'channels_first'
    else:
        reshape_dim = input_shape
        channel_order = 'channels_last'

    model_1 = Sequential()

    if batchNorm:
        model_1.add(BatchNormalization(axis=1, input_shape=reshape_dim))

    model_1.add(Conv2D(int(10 * filter_density),
                       (3, 7),
                       padding="valid",
                       input_shape=reshape_dim,
                       data_format=channel_order,
                       activation='relu'))
    model_1.add(MaxPooling2D(pool_size=(3, 1),
                             padding='valid',
                             data_format=channel_order))

    model_1.add(Conv2D(int(20 * filter_density),
                       (3, 3),
                       padding="valid",
                       data_format=channel_order,
                       activation='relu'))
    model_1.add(MaxPooling2D(pool_size=(3, 1),
                             padding='valid',
                             data_format=channel_order))

    if dropout:
        model_1.add(Dropout(dropout))

    model_1.add(Flatten())

    model_1.add(Dense(units=256, activation=dense_activation))

    if dropout:
        model_1.add(Dropout(dropout))

    model_1.add(Dense(1, activation='sigmoid'))

    optimizer = Adam()

    model_1.compile(loss='binary_crossentropy',
                    optimizer=optimizer,
                    metrics=['accuracy'])

    print(model_1.summary())

    return model_1


def createModel_schluter_valid(input, num_filter, height_filter, width_filter, filter_density, pool_n_row,
                               pool_n_col, dropout):
    """
    original Schluter relu activation, no dropout
    :param input:
    :param num_filter:
    :param height_filter:
    :param width_filter:
    :param filter_density:
    :param pool_n_row:
    :param pool_n_col:
    :param dropout:
    :return:
    """

    x = ZeroPadding2D(padding=(0, int(width_filter/2)),  data_format="channels_first")(input)

    x = Conv2D(int(num_filter * filter_density), (height_filter, width_filter), padding="valid",
               data_format="channels_first",
               activation='relu')(x)

    output_shape = K.int_shape(x)

    if pool_n_row == 'all' and pool_n_col == 'all':
        x = MaxPooling2D(pool_size=(output_shape[2], output_shape[3]), padding='same', data_format="channels_first")(x)
    elif pool_n_row == 'all' and pool_n_col != 'all':
        x = MaxPooling2D(pool_size=(output_shape[2], pool_n_col), padding='same', data_format="channels_first")(x)
    elif pool_n_row != 'all' and pool_n_col == 'all':
        x = MaxPooling2D(pool_size=(pool_n_row, output_shape[3]), padding='same', data_format="channels_first")(x)
    else:
        x = MaxPooling2D(pool_size=(pool_n_row, pool_n_col), padding='same', data_format="channels_first")(x)
    x = Dropout(dropout)(x)
    x = Flatten()(x)

    return x


def timbral_layer_schluter_valid(filter_density_layer1, pool_n_row, pool_n_col, dropout, input_dim):
    """create multi-filters layer"""
    reshape_dim = (1, input_dim[0], input_dim[1])

    input = Input(shape=reshape_dim)

    x_1 = createModel_schluter_valid(input, 32, 50, 1, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    x_2 = createModel_schluter_valid(input, 16, 50, 5, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    x_3 = createModel_schluter_valid(input, 8, 50, 10, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    x_4 = createModel_schluter_valid(input, 32, 70, 1, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    x_5 = createModel_schluter_valid(input, 16, 70, 5, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    x_6 = createModel_schluter_valid(input, 8, 70, 10, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    merged = concatenate([x_1, x_2, x_3, x_4, x_5, x_6])


    return input, merged


def model_layer2_schluter(input, merged):
    """
    create the model with the multi-filters merged input
    :param input:
    :param merged:
    :param filter_density_layer2:
    :param dropout:
    :return:
    """

    # conv 2 layers

    merged = Dense(29, activation='softmax')(merged)

    model_merged = Model(inputs=input, outputs=merged)

    optimizer = Adam()

    model_merged.compile(loss='categorical_crossentropy',
                         optimizer= optimizer,
                         metrics=['accuracy'])

    print(model_merged.summary())

    return model_merged


def jordi_model_schluter(filter_density_1,
                        pool_n_row, pool_n_col,
                        dropout, input_shape):
    """
    Schluter model configuration
    :param filter_density_1:
    :param filter_density_2:
    :param pool_n_row:
    :param pool_n_col:
    :param dropout:
    :param input_shape:
    :param dim:
    :return:
    """
    inputs, merged = timbral_layer_schluter_valid(filter_density_layer1=filter_density_1,
                                                  pool_n_row=pool_n_row,
                                                  pool_n_col=pool_n_col,
                                                  dropout=dropout,
                                                  input_dim=input_shape)

    model = model_layer2_schluter(input=inputs,
                                  merged=merged)

    return model


def model_train_validation(model_0,
                           batch_size,
                           patience,
                           input_shape,
                           path_feature_data,
                           indices_train,
                           Y_train,
                           indices_validation,
                           Y_validation,
                           file_path_model,
                           filename_log,
                           channel):

    """
    train the model with validation early stopping
    :param model_0:
    :param batch_size:
    :param patience:
    :param input_shape:
    :param path_feature_data:
    :param indices_train:
    :param Y_train:
    :param indices_validation:
    :param Y_validation:
    :param file_path_model:
    :param filename_log:
    :return:
    """

    callbacks = [ModelCheckpoint(file_path_model, monitor='val_loss', verbose=0, save_best_only=True),
                 EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
                 CSVLogger(filename=filename_log, separator=';')]

    print("start training with validation...")

    steps_per_epoch_train = int(np.ceil(len(indices_train) / batch_size))
    steps_per_epoch_val = int(np.ceil(len(indices_validation) / batch_size))

    generator_train = generator(path_feature_data=path_feature_data,
                                indices=indices_train,
                                number_of_batches=steps_per_epoch_train,
                                file_size=batch_size,
                                input_shape=input_shape,
                                labels=Y_train,
                                multi_inputs=False,
                                channel=channel)
    generator_val = generator(path_feature_data=path_feature_data,
                              indices=indices_validation,
                              number_of_batches=steps_per_epoch_val,
                              file_size=batch_size,
                              input_shape=input_shape,
                              labels=Y_validation,
                              multi_inputs=False,
                              channel=channel)

    model_0.fit_generator(generator=generator_train,
                          steps_per_epoch=steps_per_epoch_train,
                          epochs=500,
                          validation_data=generator_val,
                          validation_steps=steps_per_epoch_val,
                          callbacks=callbacks,
                          verbose=2)


def model_switcher(filter_density,
                   dropout,
                   input_shape):

    model_0 = jordi_model_schluter(filter_density_1=filter_density,
                                   pool_n_row=2,
                                   pool_n_col='all',
                                   dropout=dropout,
                                   input_shape=input_shape)

    return model_0


def train_model_validation(filename_train_validation_set,
                           filename_labels_train_validation_set,
                           filter_density,
                           dropout,
                           input_shape,
                           file_path_model,
                           filename_log,
                           channel=1):
    """
    train model with validation
    """

    filenames_train, Y_train, \
    filenames_validation, Y_validation, \
    filenames_features, Y_train_validation = \
        load_data(filename_labels_train_validation_set)

    model_0 = model_switcher(filter_density,dropout,input_shape)

    batch_size = 256
    patience = 15

    model_train_validation(model_0,
                           batch_size,
                           patience,
                           input_shape,
                           filename_train_validation_set,
                           filenames_train, Y_train,
                           filenames_validation, Y_validation,
                           file_path_model,
                           filename_log,
                           channel)
