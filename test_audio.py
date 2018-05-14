import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from scipy.io import wavfile
from torch.autograd import Variable

from data_preprocess import slice_signal, window_size, sample_rate
from model import Generator
from utils import de_emphasis, pre_emphasis

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Single Audio Enhancement')
    parser.add_argument('--file_name', type=str, required=True, help='audio file name')
    parser.add_argument('--epoch_name', type=str, required=True, help='generator epoch name')

    opt = parser.parse_args()
    FILE_NAME = opt.file_name
    EPOCH_NAME = opt.epoch_name

    generator = Generator()
    generator.load_state_dict(torch.load('epochs/' + EPOCH_NAME, map_location='cpu'))
    if torch.cuda.is_available():
        generator.cuda()

    noisy_slices = slice_signal(FILE_NAME, window_size, 1, sample_rate)
    enhanced_speech = []
    for noisy_slice in noisy_slices:
        z = nn.init.normal(torch.Tensor(1, 1024, 8))
        noisy_slice = torch.from_numpy(pre_emphasis(noisy_slice[np.newaxis, np.newaxis, :])).type(torch.FloatTensor)
        if torch.cuda.is_available():
            noisy_slice = noisy_slice.cuda()
        noisy_slice = Variable(noisy_slice)
        generated_speech = generator(noisy_slice, z).data.cpu().numpy()
        generated_speech = de_emphasis(generated_speech, emph_coeff=0.95)
        generated_speech = generated_speech.reshape(-1)
        enhanced_speech.append(generated_speech)

    enhanced_speech = np.array(enhanced_speech)
    file_name = os.path.join(os.path.dirname(FILE_NAME),
                             'enhanced_{}.wav'.format(os.path.basename(FILE_NAME).split('.')[0]))
    wavfile.write(file_name, sample_rate, enhanced_speech.T)
