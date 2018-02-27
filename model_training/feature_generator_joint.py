import numpy as np
import h5py

def shuffle_filenames_labels_in_unison(filenames,
                                       labels_syllable,
                                       labels_phoneme,
                                       sample_weights_syllable,
                                       sample_weights_phoneme):
    assert len(filenames) == len(labels_syllable)
    assert len(filenames) == len(sample_weights_syllable)
    assert len(filenames) == len(labels_phoneme)
    assert len(filenames) == len(sample_weights_phoneme)
    p = np.random.permutation(len(filenames))
    return filenames[p], labels_syllable[p], labels_phoneme[p], sample_weights_syllable[p], sample_weights_phoneme[p]

def generator(path_feature_data,
              indices,
              number_of_batches,
              file_size,
              input_shape,
              labels_syllable=None,
              labels_phoneme=None,
              sample_weights_syllable=None,
              sample_weights_phoneme=None,
              shuffle=True,
              channel=1):

    f = h5py.File(path_feature_data, 'r')
    indices_copy = np.array(indices[:], np.int64)

    if labels_syllable is not None:
        labels_syllable_copy = np.copy(labels_syllable)
    else:
        labels_syllable_copy = np.zeros((len(indices_copy), ))

    if labels_phoneme is not None:
        labels_phoneme_copy = np.copy(labels_phoneme)
    else:
        labels_phoneme_copy = np.zeros((len(indices_copy), ))

    if sample_weights_syllable is not None:
        sample_weights_syllable_copy = np.copy(sample_weights_syllable)
    else:
        sample_weights_syllable_copy = np.ones((len(indices_copy), ))

    if sample_weights_phoneme is not None:
        sample_weights_phoneme_copy = np.copy(sample_weights_phoneme)
    else:
        sample_weights_phoneme_copy = np.ones((len(indices_copy), ))

    counter = 0

    while True:
        idx_start = file_size * counter
        idx_end = file_size * (counter + 1)

        batch_indices = indices_copy[idx_start:idx_end]
        index_sort = np.argsort(batch_indices)

        y_batch_syllable_tensor = labels_syllable_copy[idx_start:idx_end][index_sort]
        y_batch_phoneme_tensor = labels_phoneme_copy[idx_start:idx_end][index_sort]

        sample_weights_batch_syllable_tensor = sample_weights_syllable_copy[idx_start:idx_end][index_sort]
        sample_weights_batch_phoneme_tensor = sample_weights_phoneme_copy[idx_start:idx_end][index_sort]

        if channel == 1:
            X_batch_tensor = f['feature_all'][batch_indices[index_sort],:,:]
        else:
            X_batch_tensor = f['feature_all'][batch_indices[index_sort], :, :, :]
        if channel == 1:
            X_batch_tensor = np.expand_dims(X_batch_tensor, axis=1)

        counter += 1

        yield X_batch_tensor, [y_batch_syllable_tensor, y_batch_phoneme_tensor], [sample_weights_batch_syllable_tensor, sample_weights_batch_phoneme_tensor]

        if counter >= number_of_batches:
            counter = 0
            if shuffle:
                indices_copy, \
                labels_syllable_copy, \
                labels_phoneme_copy, \
                sample_weights_syllable_copy, \
                sample_weights_phoneme_copy = \
                    shuffle_filenames_labels_in_unison(indices_copy,
                                                       labels_syllable_copy,
                                                       labels_phoneme_copy,
                                                       sample_weights_syllable_copy,
                                                       sample_weights_phoneme_copy)