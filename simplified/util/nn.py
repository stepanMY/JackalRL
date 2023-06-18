import torch
import torch.nn as nn


class ConvResBlock(nn.Module):
    def __init__(self, input_dim, channel, stride, padding):
        """
        Convolutional block with residual connections

        :param input_dim: int, dimension of input
        :param channel: int, dimension of output channel
        :param stride: int, stride in convolutions
        :param padding: int, padding size in convolutions
        """
        super(ConvResBlock, self).__init__()
        self.input_dim = input_dim
        self.channel = channel
        self.stride = stride
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_dim, channel//2, kernel_size=3, stride=stride, padding=padding),
            nn.ReLU(),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(channel//2, channel, kernel_size=3, stride=1, padding=padding),
            nn.ReLU(),
        )
        if self.stride == 2:
            self.bridge = nn.Conv2d(input_dim, channel, kernel_size=1, stride=2, padding=0)
        elif self.input_dim != self.channel:
            self.bridge = nn.Conv2d(input_dim, channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        Forward pass

        :param x: torch.tensor, input
        :return: torch.tensor
        """
        x1 = self.conv_block1(x)
        if self.stride != 1 or self.input_dim != self.channel:
            x2 = self.conv_block2(x1) + self.bridge(x)
        else:
            x2 = self.conv_block2(x1) + x
        return x2


class DeConvResBlock(nn.Module):
    def __init__(self, input_dim, channel, stride, padding, skipcon_dim):
        """
        DeConvolutional block with residual connections

        :param input_dim: int, dimension of input
        :param channel: int, dimension of output channel
        :param stride: int, stride in convolutions
        :param padding: int, padding size in convolutions
        :param skipcon_dim: int, size of input in skip-connections
        """
        super(DeConvResBlock, self).__init__()
        self.input_dim = input_dim
        self.channel = channel
        self.stride = stride
        self.deconv_block1 = nn.Sequential(
            nn.ConvTranspose2d(input_dim, channel//2, kernel_size=3, stride=stride, padding=padding),
            nn.ReLU(),
        )
        self.deconv_block2 = nn.Sequential(
            nn.ConvTranspose2d(channel, channel, kernel_size=3, stride=1, padding=padding),
            nn.ReLU(),
        )
        self.deconv_skipcon = nn.ConvTranspose2d(skipcon_dim, channel//2, kernel_size=1, stride=1, padding=0)
        if self.stride == 2 and self.input_dim != self.channel:
            self.bridge = nn.ConvTranspose2d(input_dim, channel, kernel_size=1, stride=2, padding=0)

    def forward(self, x, skipcon):
        """
        Forward pass

        :param x: torch.tensor, input
        :param skipcon: torch.tensor, skip-connection value
        :return: torch.tensor
        """
        x1 = self.deconv_block1(x)
        skipcon_value = self.deconv_skipcon(skipcon)
        concat = torch.cat([x1, skipcon_value], dim=1)
        if self.stride == 2 and self.input_dim != self.channel:
            x2 = self.deconv_block2(concat) + self.bridge(x)
        else:
            x2 = self.deconv_block2(concat) + x
        return x2


class ResUnet(nn.Module):
    def __init__(self, input_dim, output_dim, field_shape):
        """
        Unet like architecture with residual blocks

        :param input_dim: int, dimension of input
        :param output_dim: int, dimension of output channel
        :param field_shape: tuple, dimensions of Jackal field
        """
        super(ResUnet, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(input_dim, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.outer_convresblock = ConvResBlock(256, 256, 1, 1)
        self.strided_convresblock = ConvResBlock(256, 256, 2, 1)
        self.inner_convresblock = ConvResBlock(256, 320, 1, 1)
        self.inner_deconvresblock = DeConvResBlock(320, 320, 1, 1, 256)
        self.strided_deconvresblock = DeConvResBlock(320, 256, 2, 1, 256)
        self.outer_deconvresblock = DeConvResBlock(256, 256, 1, 1, 256)
        self.final_conv = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.classifier_layer = nn.Linear(field_shape[0] * field_shape[1], output_dim)

    def forward(self, x):
        x1 = self.input_layer(x)
        x2 = self.outer_convresblock(x1)
        x3 = self.strided_convresblock(x2)
        x4 = self.inner_convresblock(x3)
        x5 = self.inner_deconvresblock(x4, x3)
        x6 = self.strided_deconvresblock(x5, x2)
        x7 = self.outer_deconvresblock(x6, x1)
        x8 = self.final_conv(x7)
        x8_flatten = torch.flatten(x8, 1)
        output = self.classifier_layer(x8_flatten)
        return output
