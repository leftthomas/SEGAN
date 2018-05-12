import os

import torch
import torch.nn as nn
from scipy.io import wavfile
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data_generator import AudioSampleGenerator
from model import Generator, Discriminator
from utils import de_emphasis, split_pair_to_vars

"""
Here we define the discriminator and generator for SEGAN.
After definition of each modules, run the training.
"""

# define folders for data
data_path = 'data'
clean_train_foldername = 'clean_trainset_56spk_wav'
noisy_train_foldername = 'noisy_trainset_56spk_wav'
out_clean_train_fdrnm = 'clean_trainset_wav_16k'
out_noisy_train_fdrnm = 'noisy_trainset_wav_16k'
ser_data_fdrnm = 'ser_data'  # serialized data
gen_data_fdrnm = 'gen_data_v5'  # folder for saving generated data
model_fdrnm = 'epochs'  # folder for saving models

# create folder for generated data
gen_data_path = os.path.join(os.getcwd(), gen_data_fdrnm)
if not os.path.exists(gen_data_path):
    os.makedirs(gen_data_path)

# create folder for model checkpoints
models_path = os.path.join(os.getcwd(), model_fdrnm)
if not os.path.exists(models_path):
    os.makedirs(models_path)

# batch size
batch_size = 50
d_learning_rate = 0.0001
g_learning_rate = 0.0001
g_lambda = 100  # regularizer for generator
sample_rate = 16000
torch.cuda.manual_seed_all(5)  # fix random seed

# create D and G instances
discriminator = Discriminator().cuda()  # use GPU
generator = Generator(batch_size).cuda()

# This is how you define a data loader
sample_generator = AudioSampleGenerator(os.path.join(data_path, ser_data_fdrnm))
random_data_loader = DataLoader(
    dataset=sample_generator,
    batch_size=batch_size,  # specified batch size here
    shuffle=True,
    num_workers=4,
    drop_last=True,  # drop the last batch that cannot be divided by batch_size
    pin_memory=True)

# generate reference batch
ref_batch_pairs = sample_generator.reference_batch(batch_size)
ref_batch_var, ref_clean_var, ref_noisy_var = split_pair_to_vars(ref_batch_pairs)

# test samples for generation
test_noise_filenames, fixed_test_noise = sample_generator.fixed_test_audio(batch_size)
fixed_test_noise = Variable(torch.from_numpy(fixed_test_noise)).cuda()
print('Test samples loaded')

# optimizers
g_optimizer = optim.RMSprop(generator.parameters(), lr=g_learning_rate)
d_optimizer = optim.RMSprop(discriminator.parameters(), lr=d_learning_rate)

print('Starting Training...')
for epoch in range(86):
    for i, sample_batch_pairs in enumerate(random_data_loader):
        # using the sample batch pair, split into
        # batch of combined pairs, clean signals, and noisy signals
        batch_pairs_var, clean_batch_var, noisy_batch_var = split_pair_to_vars(sample_batch_pairs)

        # latent vector - normal distribution
        z = Variable(nn.init.normal(torch.Tensor(batch_size, 1024, 8))).cuda()

        # TRAIN D to recognize clean audio as clean
        # training batch pass
        discriminator.zero_grad()
        outputs = discriminator(batch_pairs_var, ref_batch_var)  # output : [40 x 1 x 8]
        clean_loss = torch.mean((outputs - 1.0) ** 2)  # L2 loss - we want them all to be 1
        clean_loss.backward()

        # TRAIN D to recognize generated audio as noisy
        generated_outputs = generator(noisy_batch_var, z)
        disc_in_pair = torch.cat((generated_outputs, noisy_batch_var), dim=1)
        outputs = discriminator(disc_in_pair, ref_batch_var)
        noisy_loss = torch.mean(outputs ** 2)  # L2 loss - we want them all to be 0
        noisy_loss.backward()

        # d_loss = clean_loss + noisy_loss
        d_optimizer.step()  # update parameters

        # TRAIN G so that D recognizes G(z) as real
        generator.zero_grad()
        generated_outputs = generator(noisy_batch_var, z)
        gen_noise_pair = torch.cat((generated_outputs, noisy_batch_var), dim=1)
        outputs = discriminator(gen_noise_pair, ref_batch_var)

        g_loss_ = 0.5 * torch.mean((outputs - 1.0) ** 2)
        # L1 loss between generated output and clean sample
        l1_dist = torch.abs(torch.add(generated_outputs, torch.neg(clean_batch_var)))
        g_cond_loss = g_lambda * torch.mean(l1_dist)  # conditional loss
        g_loss = g_loss_ + g_cond_loss

        # backprop + optimize
        g_loss.backward()
        g_optimizer.step()

        # print message per 10 steps
        if (i + 1) % 10 == 0:
            print('Epoch {}, Step {}, d_clean_loss {}, d_noisy_loss {}, g_loss {}, g_loss_cond {}'
                .format(
                epoch + 1, i + 1, clean_loss.data[0],
                noisy_loss.data[0], g_loss.data[0], g_cond_loss.data[0]))
            # Functions below print various information about the network. Uncomment to use.
            # print('Weight for latent variable z : {}'.format(z))
            # print('Generated Outputs : {}'.format(generated_outputs))
            # print('Encoding 8th layer weight: {}'.format(generator.module.enc8.weight))

        # save sampled audio at the beginning of each epoch
        if i == 0:
            fake_speech = generator(fixed_test_noise, z)
            fake_speech_data = fake_speech.data.cpu().numpy()  # convert to numpy array
            fake_speech_data = de_emphasis(fake_speech_data, emph_coeff=0.95)

            for idx in range(4):  # select four samples
                generated_sample = fake_speech_data[idx]
                filepath = os.path.join(
                    gen_data_path, '{}_e{}.wav'.format(test_noise_filenames[idx], epoch + 1))
                wavfile.write(filepath, sample_rate, generated_sample.T)

    # save the model parameters for each epoch
    g_path = os.path.join(models_path, 'generator-{}.pkl'.format(epoch + 1))
    d_path = os.path.join(models_path, 'discriminator-{}.pkl'.format(epoch + 1))
    torch.save(generator.state_dict(), g_path)
    torch.save(discriminator.state_dict(), d_path)
print('Finished Training!')
