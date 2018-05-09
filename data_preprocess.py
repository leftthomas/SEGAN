import os
import subprocess
import librosa
import numpy as np
import time

data_path = 'data'  # the base folder for dataset
clean_train_foldername = 'clean_trainset_56spk_wav'  # where original clean train data exist
noisy_train_foldername = 'noisy_trainset_56spk_wav'  # where original noisy train data exist
out_clean_train_fdrnm = 'clean_trainset_wav_16k'  # clean preprocessed data folder
out_noisy_train_fdrnm = 'noisy_trainset_wav_16k'  # noisy preprocessed data folder
ser_data_fdrnm = 'ser_data'  # serialized data folder


def data_verify():
    """
    Verifies the length of each data after preprocessing.
    """
    ser_data_path = os.path.join(data_path, ser_data_fdrnm)
    for dirname, dirs, files in os.walk(ser_data_path):
        for filename in files:
            data_pair = np.load(os.path.join(dirname, filename))
            if data_pair.shape[1] != 16384:
                print('Snippet length not 16384 : {} instead'.format(data_pair.shape[1]))
                break


def downsample_16k():
    """
    Convert all audio files to have sampling rate 16k.
    """
    # clean training sets
    if not os.path.exists(os.path.join(data_path, out_clean_train_fdrnm)):
        os.makedirs(os.path.join(data_path, out_clean_train_fdrnm))

    for dirname, dirs, files in os.walk(os.path.join(data_path, clean_train_foldername)):
        for filename in files:
            input_filepath = os.path.abspath(os.path.join(dirname, filename))
            output_folderpath = os.path.join(data_path, out_clean_train_fdrnm)
            # use sox to down-sample to 16k
            print('Downsampling : {}'.format(input_filepath))
            completed_process = subprocess.run(
                    'sox {} -r 16k {}'
                    .format(input_filepath, os.path.join(output_folderpath, filename)),
                    shell=True, check=True)

    # noisy training sets
    if not os.path.exists(os.path.join(data_path, out_noisy_train_fdrnm)):
        os.makedirs(os.path.join(data_path, out_noisy_train_fdrnm))

    for dirname, dirs, files in os.walk(os.path.join(data_path, noisy_train_foldername)):
        for filename in files:
            input_filepath = os.path.abspath(os.path.join(dirname, filename))
            output_folderpath = os.path.join(data_path, out_noisy_train_fdrnm)
            # use sox to down-sample to 16k
            print('Processing : {}'.format(input_filepath))
            completed_process = subprocess.run(
                    'sox {} -r 16k {}'
                    .format(input_filepath, os.path.join(output_folderpath, filename)),
                    shell=True, check=True)


def slice_signal(filepath, window_size, stride, sample_rate):
    """
    Helper function for slicing the audio file
    by window size with [stride] percent overlap (default 50%).
    """
    wav, sr = librosa.load(filepath, sr=sample_rate)
    n_samples = wav.shape[0]  # contains simple amplitudes
    hop = int(window_size * stride)
    slices = []
    for end_idx in range(window_size, len(wav), hop):
        start_idx = end_idx - window_size
        slice_sig = wav[start_idx:end_idx]
        slices.append(slice_sig)
    return slices


def process_and_serialize():
    """
    Serialize the sliced signals and save on separate folder.
    """
    start_time = time.time()  # measure the time
    window_size = 2 ** 14  # about 1 second of samples
    dst_folder = os.path.join(data_path, ser_data_fdrnm)
    sample_rate = 16000
    stride = 0.5

    if not os.path.exists(dst_folder):
        print('Creating new destination folder for new data')
        os.makedirs(dst_folder)

    # the path for source data (16k downsampled)
    clean_data_path = os.path.join(data_path, out_clean_train_fdrnm)
    noisy_data_path = os.path.join(data_path, out_noisy_train_fdrnm)

    # walk through the path, slice the audio file, and save the serialized result
    for dirname, dirs, files in os.walk(clean_data_path):
        if len(files) == 0:
            continue
        for filename in files:
            print('Splitting : {}'.format(filename))
            clean_filepath = os.path.join(clean_data_path, filename)
            noisy_filepath = os.path.join(noisy_data_path, filename)

            # slice both clean signal and noisy signal
            clean_sliced = slice_signal(clean_filepath, window_size, stride, sample_rate)
            noisy_sliced = slice_signal(noisy_filepath, window_size, stride, sample_rate)

            # serialize - file format goes [origial_file]_[slice_number].npy
            # ex) p293_154.wav_5.npy denotes 5th slice of p293_154.wav file
            for idx, slice_tuple in enumerate(zip(clean_sliced, noisy_sliced)):
                pair = np.array([slice_tuple[0], slice_tuple[1]])
                np.save(os.path.join(dst_folder, '{}_{}'.format(filename, idx)), arr=pair)

    # measure the time it took to process
    end_time = time.time()
    print('Total elapsed time for prerpocessing : {}'.format(end_time - start_time))


if __name__ == '__main__':
    downsample_16k()
    process_and_serialize()  # WARNING - takes very long time
    data_verify()
