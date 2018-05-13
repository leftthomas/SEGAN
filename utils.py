import os

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils import data


def pre_emphasis(signal_batch, emph_coeff=0.95):
    """
    Pre-emphasis of higher frequencies given a batch of signal.

    Args:
        signal_batch: batch of signals, represented as numpy arrays
        emph_coeff: emphasis coefficient

    Returns:
        result: pre-emphasized signal batch
    """
    result = np.zeros(signal_batch.shape)
    for sample_idx, sample in enumerate(signal_batch):
        for ch, channel_data in enumerate(sample):
            result[sample_idx][ch] = np.append(
                channel_data[0], channel_data[1:] - emph_coeff * channel_data[:-1])
    return result


def de_emphasis(signal_batch, emph_coeff=0.95):
    """
    Deemphasis operation given a batch of signal.
    Reverts the pre-emphasized signal.

    Args:
        signal_batch: batch of signals, represented as numpy arrays
        emph_coeff: emphasis coefficient

    Returns:
        result: de-emphasized signal batch
    """
    result = np.zeros(signal_batch.shape)
    for sample_idx, sample in enumerate(signal_batch):
        for ch, channel_data in enumerate(sample):
            result[sample_idx][ch] = np.append(
                channel_data[0], channel_data[1:] + emph_coeff * channel_data[:-1])
    return result


def split_pair_to_vars(sample_batch_pair):
    """
    Splits the generated batch data and creates combination of pairs.
    Input argument sample_batch_pair consists of a batch_size number of
    [clean_signal, noisy_signal] pairs.

    This function creates three pytorch Variables - a clean_signal, noisy_signal pair,
    clean signal only, and noisy signal only.
    It goes through preemphasis preprocessing before converted into variable.

    Args:
        sample_batch_pair(torch.Tensor): batch of [clean_signal, noisy_signal] pairs
    Returns:
        batch_pairs_var(Variable): batch of pairs containing clean signal and noisy signal
        clean_batch_var(Variable): clean signal batch
        noisy_batch_var(Varialbe): noisy signal batch
    """
    # preemphasis
    sample_batch_pair = pre_emphasis(sample_batch_pair.numpy(), emph_coeff=0.95)
    batch_pairs_var = Variable(torch.from_numpy(sample_batch_pair).type(torch.FloatTensor)).cuda()  # [40 x 2 x 16384]
    clean_batch = np.stack([pair[0].reshape(1, -1) for pair in sample_batch_pair])
    clean_batch_var = Variable(torch.from_numpy(clean_batch).type(torch.FloatTensor)).cuda()
    noisy_batch = np.stack([pair[1].reshape(1, -1) for pair in sample_batch_pair])
    noisy_batch_var = Variable(torch.from_numpy(noisy_batch).type(torch.FloatTensor)).cuda()
    return batch_pairs_var, clean_batch_var, noisy_batch_var


class AudioDataset(data.Dataset):
    """
    Audio sample reader.
    Used alongside with DataLoader class to generate batches.
    see: http://pytorch.org/docs/master/data.html#torch.utils.data.Dataset
    """

    def __init__(self, data_folder_path):
        if not os.path.exists(data_folder_path):
            raise Error('The data folder does not exist!')

        # store full paths - not the actual files.
        # all files cannot be loaded up to memory due to its large size.
        # insted, we read from files upon fetching batches (see __getitem__() implementation)
        self.filepaths = [os.path.join(data_folder_path, filename)
                          for filename in os.listdir(data_folder_path)]
        self.num_data = len(self.filepaths)

    def reference_batch(self, batch_size):
        """
        Randomly selects a reference batch from dataset.
        Reference batch is used for calculating statistics for virtual batch normalization operation.

        Args:
            batch_size(int): batch size

        Returns:
            ref_batch: reference batch
        """
        ref_filenames = np.random.choice(self.filepaths, batch_size)
        ref_batch = torch.from_numpy(np.stack([np.load(f) for f in ref_filenames]))
        return ref_batch

    def fixed_test_audio(self, num_test_audio):
        """
        Randomly chosen batch for testing generated results.

        Args:
            num_test_audio(int): number of test audio.
                Must be same as batch size of training,
                otherwise it cannot go through the forward step of generator.
        """
        test_filenames = np.random.choice(self.filepaths, num_test_audio)
        test_noisy_set = [np.load(f)[1] for f in test_filenames]
        # file names of test samples
        test_basenames = [os.path.basename(fpath) for fpath in test_filenames]
        return test_basenames, np.array(test_noisy_set).reshape(num_test_audio, 1, 16384)

    def __getitem__(self, idx):
        # get item for specified index
        pair = np.load(self.filepaths[idx])
        return pair

    def __len__(self):
        return self.num_data
