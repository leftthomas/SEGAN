import torch
import torch.nn as nn
from torch.nn.modules import Module
from torch.nn.parameter import Parameter


class VirtualBatchNorm1d(Module):
    """
    Module for Virtual Batch Normalization.

    Implementation borrowed and modified from Rafael_Valle's code + help of SimonW from this discussion thread:
    https://discuss.pytorch.org/t/parameter-grad-of-conv-weight-is-none-after-virtual-batch-normalization/9036
    """

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        # batch statistics
        self.num_features = num_features
        self.eps = eps  # epsilon
        # define gamma and beta parameters
        self.gamma = Parameter(torch.normal(mean=1.0, std=0.02, size=(1, num_features, 1)))
        self.beta = Parameter(torch.zeros(1, num_features, 1))

    def get_stats(self, x):
        """
        Calculates mean and mean square for given batch x.
        Args:
            x: tensor containing batch of activations
        Returns:
            mean: mean tensor over features
            mean_sq: squared mean tensor over features
        """
        mean = x.mean(2, keepdim=True).mean(0, keepdim=True)
        mean_sq = (x ** 2).mean(2, keepdim=True).mean(0, keepdim=True)
        return mean, mean_sq

    def forward(self, x, ref_mean, ref_mean_sq):
        """
        Forward pass of virtual batch normalization.
        Virtual batch normalization require two forward passes
        for reference batch and train batch, respectively.

        Args:
            x: input tensor
            ref_mean: reference mean tensor over features
            ref_mean_sq: reference squared mean tensor over features
        Result:
            x: normalized batch tensor
            ref_mean: reference mean tensor over features
            ref_mean_sq: reference squared mean tensor over features
        """
        mean, mean_sq = self.get_stats(x)
        if ref_mean is None or ref_mean_sq is None:
            # reference mode - works just like batch norm
            mean = mean.clone().detach()
            mean_sq = mean_sq.clone().detach()
            out = self.normalize(x, mean, mean_sq)
        else:
            # calculate new mean and mean_sq
            batch_size = x.size(0)
            new_coeff = 1. / (batch_size + 1.)
            old_coeff = 1. - new_coeff
            mean = new_coeff * mean + old_coeff * ref_mean
            mean_sq = new_coeff * mean_sq + old_coeff * ref_mean_sq
            out = self.normalize(x, mean, mean_sq)
        return out, mean, mean_sq

    def normalize(self, x, mean, mean_sq):
        """
        Normalize tensor x given the statistics.

        Args:
            x: input tensor
            mean: mean over features
            mean_sq: squared means over features

        Result:
            x: normalized batch tensor
        """
        assert mean_sq is not None
        assert mean is not None
        assert len(x.size()) == 3  # specific for 1d VBN
        if mean.size(1) != self.num_features:
            raise Exception('Mean tensor size not equal to number of features : given {}, expected {}'
                            .format(mean.size(1), self.num_features))
        if mean_sq.size(1) != self.num_features:
            raise Exception('Squared mean tensor size not equal to number of features : given {}, expected {}'
                            .format(mean_sq.size(1), self.num_features))

        std = torch.sqrt(self.eps + mean_sq - mean ** 2)
        x = x - mean
        x = x / std
        x = x * self.gamma
        x = x + self.beta
        return x

    def __repr__(self):
        return ('{name}(num_features={num_features}, eps={eps}'
                .format(name=self.__class__.__name__, **self.__dict__))


class Generator(nn.Module):
    """G"""

    def __init__(self):
        super().__init__()
        # encoder gets a noisy signal as input [B x 1 x 16384]
        self.enc1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=32, stride=2, padding=15)  # [B x 16 x 8192]
        self.enc1_nl = nn.PReLU()
        self.enc2 = nn.Conv1d(16, 32, 32, 2, 15)  # [B x 32 x 4096]
        self.enc2_nl = nn.PReLU()
        self.enc3 = nn.Conv1d(32, 32, 32, 2, 15)  # [B x 32 x 2048]
        self.enc3_nl = nn.PReLU()
        self.enc4 = nn.Conv1d(32, 64, 32, 2, 15)  # [B x 64 x 1024]
        self.enc4_nl = nn.PReLU()
        self.enc5 = nn.Conv1d(64, 64, 32, 2, 15)  # [B x 64 x 512]
        self.enc5_nl = nn.PReLU()
        self.enc6 = nn.Conv1d(64, 128, 32, 2, 15)  # [B x 128 x 256]
        self.enc6_nl = nn.PReLU()
        self.enc7 = nn.Conv1d(128, 128, 32, 2, 15)  # [B x 128 x 128]
        self.enc7_nl = nn.PReLU()
        self.enc8 = nn.Conv1d(128, 256, 32, 2, 15)  # [B x 256 x 64]
        self.enc8_nl = nn.PReLU()
        self.enc9 = nn.Conv1d(256, 256, 32, 2, 15)  # [B x 256 x 32]
        self.enc9_nl = nn.PReLU()
        self.enc10 = nn.Conv1d(256, 512, 32, 2, 15)  # [B x 512 x 16]
        self.enc10_nl = nn.PReLU()
        self.enc11 = nn.Conv1d(512, 1024, 32, 2, 15)  # [B x 1024 x 8]
        self.enc11_nl = nn.PReLU()

        # decoder generates an enhanced signal
        # each decoder output are concatenated with homologous encoder output,
        # so the feature map sizes are doubled
        self.dec10 = nn.ConvTranspose1d(in_channels=2048, out_channels=512, kernel_size=32, stride=2, padding=15)
        self.dec10_nl = nn.PReLU()  # out : [B x 512 x 16] -> (concat) [B x 1024 x 16]
        self.dec9 = nn.ConvTranspose1d(1024, 256, 32, 2, 15)  # [B x 256 x 32]
        self.dec9_nl = nn.PReLU()
        self.dec8 = nn.ConvTranspose1d(512, 256, 32, 2, 15)  # [B x 256 x 64]
        self.dec8_nl = nn.PReLU()
        self.dec7 = nn.ConvTranspose1d(512, 128, 32, 2, 15)  # [B x 128 x 128]
        self.dec7_nl = nn.PReLU()
        self.dec6 = nn.ConvTranspose1d(256, 128, 32, 2, 15)  # [B x 128 x 256]
        self.dec6_nl = nn.PReLU()
        self.dec5 = nn.ConvTranspose1d(256, 64, 32, 2, 15)  # [B x 64 x 512]
        self.dec5_nl = nn.PReLU()
        self.dec4 = nn.ConvTranspose1d(128, 64, 32, 2, 15)  # [B x 64 x 1024]
        self.dec4_nl = nn.PReLU()
        self.dec3 = nn.ConvTranspose1d(128, 32, 32, 2, 15)  # [B x 32 x 2048]
        self.dec3_nl = nn.PReLU()
        self.dec2 = nn.ConvTranspose1d(64, 32, 32, 2, 15)  # [B x 32 x 4096]
        self.dec2_nl = nn.PReLU()
        self.dec1 = nn.ConvTranspose1d(64, 16, 32, 2, 15)  # [B x 16 x 8192]
        self.dec1_nl = nn.PReLU()
        self.dec_final = nn.ConvTranspose1d(32, 1, 32, 2, 15)  # [B x 1 x 16384]
        self.dec_tanh = nn.Tanh()

        # initialize weights
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights for convolution layers using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal(m.weight.data)

    def forward(self, x, z):
        """
        Forward pass of generator.

        Args:
            x: input batch (signal)
            z: latent vector
        """
        # encoding step
        e1 = self.enc1(x)
        e2 = self.enc2(self.enc1_nl(e1))
        e3 = self.enc3(self.enc2_nl(e2))
        e4 = self.enc4(self.enc3_nl(e3))
        e5 = self.enc5(self.enc4_nl(e4))
        e6 = self.enc6(self.enc5_nl(e5))
        e7 = self.enc7(self.enc6_nl(e6))
        e8 = self.enc8(self.enc7_nl(e7))
        e9 = self.enc9(self.enc8_nl(e8))
        e10 = self.enc10(self.enc9_nl(e9))
        e11 = self.enc11(self.enc10_nl(e10))
        # c = compressed feature, the 'thought vector'
        c = self.enc11_nl(e11)

        # concatenate the thought vector with latent variable
        encoded = torch.cat((c, z), dim=1)

        # decoding step
        d10 = self.dec10(encoded)
        # dx_c : concatenated with skip-connected layer's output & passed nonlinear layer
        d10_c = self.dec10_nl(torch.cat((d10, e10), dim=1))
        d9 = self.dec9(d10_c)
        d9_c = self.dec9_nl(torch.cat((d9, e9), dim=1))
        d8 = self.dec8(d9_c)
        d8_c = self.dec8_nl(torch.cat((d8, e8), dim=1))
        d7 = self.dec7(d8_c)
        d7_c = self.dec7_nl(torch.cat((d7, e7), dim=1))
        d6 = self.dec6(d7_c)
        d6_c = self.dec6_nl(torch.cat((d6, e6), dim=1))
        d5 = self.dec5(d6_c)
        d5_c = self.dec5_nl(torch.cat((d5, e5), dim=1))
        d4 = self.dec4(d5_c)
        d4_c = self.dec4_nl(torch.cat((d4, e4), dim=1))
        d3 = self.dec3(d4_c)
        d3_c = self.dec3_nl(torch.cat((d3, e3), dim=1))
        d2 = self.dec2(d3_c)
        d2_c = self.dec2_nl(torch.cat((d2, e2), dim=1))
        d1 = self.dec1(d2_c)
        d1_c = self.dec1_nl(torch.cat((d1, e1), dim=1))
        out = self.dec_tanh(self.dec_final(d1_c))
        return out


class Discriminator(nn.Module):
    """D"""

    def __init__(self):
        super().__init__()
        # D gets a noisy signal and clear signal as input [B x 2 x 16384]
        negative_slope = 0.03
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=31, stride=2, padding=15)  # [B x 32 x 8192]
        self.vbn1 = VirtualBatchNorm1d(32)
        self.lrelu1 = nn.LeakyReLU(negative_slope)
        self.conv2 = nn.Conv1d(32, 64, 31, 2, 15)  # [B x 64 x 4096]
        self.vbn2 = VirtualBatchNorm1d(64)
        self.lrelu2 = nn.LeakyReLU(negative_slope)
        self.conv3 = nn.Conv1d(64, 64, 31, 2, 15)  # [B x 64 x 2048]
        self.dropout1 = nn.Dropout()
        self.vbn3 = VirtualBatchNorm1d(64)
        self.lrelu3 = nn.LeakyReLU(negative_slope)
        self.conv4 = nn.Conv1d(64, 128, 31, 2, 15)  # [B x 128 x 1024]
        self.vbn4 = VirtualBatchNorm1d(128)
        self.lrelu4 = nn.LeakyReLU(negative_slope)
        self.conv5 = nn.Conv1d(128, 128, 31, 2, 15)  # [B x 128 x 512]
        self.vbn5 = VirtualBatchNorm1d(128)
        self.lrelu5 = nn.LeakyReLU(negative_slope)
        self.conv6 = nn.Conv1d(128, 256, 31, 2, 15)  # [B x 256 x 256]
        self.dropout2 = nn.Dropout()
        self.vbn6 = VirtualBatchNorm1d(256)
        self.lrelu6 = nn.LeakyReLU(negative_slope)
        self.conv7 = nn.Conv1d(256, 256, 31, 2, 15)  # [B x 256 x 128]
        self.vbn7 = VirtualBatchNorm1d(256)
        self.lrelu7 = nn.LeakyReLU(negative_slope)
        self.conv8 = nn.Conv1d(256, 512, 31, 2, 15)  # [B x 512 x 64]
        self.vbn8 = VirtualBatchNorm1d(512)
        self.lrelu8 = nn.LeakyReLU(negative_slope)
        self.conv9 = nn.Conv1d(512, 512, 31, 2, 15)  # [B x 512 x 32]
        self.dropout3 = nn.Dropout()
        self.vbn9 = VirtualBatchNorm1d(512)
        self.lrelu9 = nn.LeakyReLU(negative_slope)
        self.conv10 = nn.Conv1d(512, 1024, 31, 2, 15)  # [B x 1024 x 16]
        self.vbn10 = VirtualBatchNorm1d(1024)
        self.lrelu10 = nn.LeakyReLU(negative_slope)
        self.conv11 = nn.Conv1d(1024, 2048, 31, 2, 15)  # [B x 2048 x 8]
        self.vbn11 = VirtualBatchNorm1d(2048)
        self.lrelu11 = nn.LeakyReLU(negative_slope)
        # 1x1 size kernel for dimension and parameter reduction
        self.conv_final = nn.Conv1d(2048, 1, kernel_size=1, stride=1)  # [B x 1 x 8]
        self.lrelu_final = nn.LeakyReLU(negative_slope)
        self.fully_connected = nn.Linear(in_features=8, out_features=1)  # [B x 1]
        self.sigmoid = nn.Sigmoid()

        # initialize weights
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights for convolution layers using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal(m.weight.data)

    def forward(self, x, ref_x):
        """
        Forward pass of discriminator.

        Args:
            x: input batch (signal)
            ref_x: reference input batch for virtual batch norm
        """
        # reference pass
        ref_x = self.conv1(ref_x)
        ref_x, mean1, meansq1 = self.vbn1(ref_x, None, None)
        ref_x = self.lrelu1(ref_x)
        ref_x = self.conv2(ref_x)
        ref_x, mean2, meansq2 = self.vbn2(ref_x, None, None)
        ref_x = self.lrelu2(ref_x)
        ref_x = self.conv3(ref_x)
        ref_x = self.dropout1(ref_x)
        ref_x, mean3, meansq3 = self.vbn3(ref_x, None, None)
        ref_x = self.lrelu3(ref_x)
        ref_x = self.conv4(ref_x)
        ref_x, mean4, meansq4 = self.vbn4(ref_x, None, None)
        ref_x = self.lrelu4(ref_x)
        ref_x = self.conv5(ref_x)
        ref_x, mean5, meansq5 = self.vbn5(ref_x, None, None)
        ref_x = self.lrelu5(ref_x)
        ref_x = self.conv6(ref_x)
        ref_x = self.dropout2(ref_x)
        ref_x, mean6, meansq6 = self.vbn6(ref_x, None, None)
        ref_x = self.lrelu6(ref_x)
        ref_x = self.conv7(ref_x)
        ref_x, mean7, meansq7 = self.vbn7(ref_x, None, None)
        ref_x = self.lrelu7(ref_x)
        ref_x = self.conv8(ref_x)
        ref_x, mean8, meansq8 = self.vbn8(ref_x, None, None)
        ref_x = self.lrelu8(ref_x)
        ref_x = self.conv9(ref_x)
        ref_x = self.dropout3(ref_x)
        ref_x, mean9, meansq9 = self.vbn9(ref_x, None, None)
        ref_x = self.lrelu9(ref_x)
        ref_x = self.conv10(ref_x)
        ref_x, mean10, meansq10 = self.vbn10(ref_x, None, None)
        ref_x = self.lrelu10(ref_x)
        ref_x = self.conv11(ref_x)
        ref_x, mean11, meansq11 = self.vbn11(ref_x, None, None)
        # further pass no longer needed

        # train pass
        x = self.conv1(x)
        x, _, _ = self.vbn1(x, mean1, meansq1)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x, _, _ = self.vbn2(x, mean2, meansq2)
        x = self.lrelu2(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        x, _, _ = self.vbn3(x, mean3, meansq3)
        x = self.lrelu3(x)
        x = self.conv4(x)
        x, _, _ = self.vbn4(x, mean4, meansq4)
        x = self.lrelu4(x)
        x = self.conv5(x)
        x, _, _ = self.vbn5(x, mean5, meansq5)
        x = self.lrelu5(x)
        x = self.conv6(x)
        x = self.dropout2(x)
        x, _, _ = self.vbn6(x, mean6, meansq6)
        x = self.lrelu6(x)
        x = self.conv7(x)
        x, _, _ = self.vbn7(x, mean7, meansq7)
        x = self.lrelu7(x)
        x = self.conv8(x)
        x, _, _ = self.vbn8(x, mean8, meansq8)
        x = self.lrelu8(x)
        x = self.conv9(x)
        x = self.dropout3(x)
        x, _, _ = self.vbn9(x, mean9, meansq9)
        x = self.lrelu9(x)
        x = self.conv10(x)
        x, _, _ = self.vbn10(x, mean10, meansq10)
        x = self.lrelu10(x)
        x = self.conv11(x)
        x, _, _ = self.vbn11(x, mean11, meansq11)
        x = self.lrelu11(x)
        x = self.conv_final(x)
        x = self.lrelu_final(x)
        # reduce down to a scalar value
        x = torch.squeeze(x)
        x = self.fully_connected(x)
        return self.sigmoid(x)
