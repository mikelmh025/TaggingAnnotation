"""
This file defines the core research contribution
"""
import torch
from torch import nn



					 


def conv_layer(in_chanel, out_chanel, kernel_size, stride=1, pad=0):
    return nn.Sequential(nn.Conv2d(in_chanel, out_chanel, kernel_size=kernel_size, stride=stride, padding=pad),
                         nn.BatchNorm2d(out_chanel), nn.ReLU())


class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, mode=1):
        """
        mfm
        :param in_channels: in channel
        :param out_channels: out channel
        :param kernel_size: conv kernel size
        :param stride: conv stride
        :param padding: conv padding
        :param mode: 1: Conv2d  2: Linear
        """
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if mode == 1:
            self.filter = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=kernel_size, stride=stride,
                                    padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2 * out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])


class ResidualBlock(nn.Module):
    """
    残差网络
    """

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = mfm(in_channels, out_channels)
        self.conv2 = mfm(in_channels, out_channels)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + res
        return out

    @staticmethod
    def make_layer(num_blocks, channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(ResidualBlock(channels, channels))
        return nn.Sequential(*layers)


class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv = mfm(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x

class resnet_var(nn.Module):

	def __init__(self, res=256, fc_out=136):
		super(resnet_var, self).__init__()

		self.resnet = nn.Sequential(
			nn.Conv2d(3, 4, kernel_size=7, stride=2, padding=3),  # 1. (batch, 4, 32, 32)
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 2. (batch, 4, 16, 16)
			group(4, 8, kernel_size=3, stride=1, padding=1),  # 3. (batch, 8, 16, 16)
			ResidualBlock.make_layer(8, channels=8),  # 4. (batch, 8, 16, 16)
			group(8, 16, kernel_size=3, stride=1, padding=1),  # 5. (batch, 16, 16, 16)
			ResidualBlock.make_layer(8, channels=16),  # 6. (batch, 16, 16, 16)
			group(16, 64, kernel_size=3, stride=1, padding=1),  # 7. (batch, 64, 16, 16)
			ResidualBlock.make_layer(8, channels=64),  # 8. (batch, 64, 16, 16)
			group(64, 99, kernel_size=3, stride=1, padding=1),  # 9. (batch, params_cnt, 16, 16)
			ResidualBlock.make_layer(4, channels=99),  # 10. (batch, params_cnt, 16, 16)
			nn.Dropout(0.5),
		)
		fc_in = int(99 * res * res /16)
		self.fc = nn.Sequential(
							nn.Linear(fc_in,512),
							nn.Linear(512,256),
							nn.Linear(256,256),
							nn.Linear(256,fc_out),
		)

	def forward(self, x):
		x = self.resnet(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x

def resnet_256(fc_out=136):
    return resnet_var(res=256,fc_out=fc_out)


if __name__ == '__main__':
    model = resnet_256(10).cuda()
    input = torch.randn(1, 3, 256, 256).cuda()
    output = model(input)
    print(output.shape)
    print(model)