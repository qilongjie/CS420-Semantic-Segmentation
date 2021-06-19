import torch
import torch.nn as nn

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        # 下采样
        def LinkConv(in_channels, middle_channels, out_channels, kernel_size, padding):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=middle_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(num_features=middle_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=middle_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(),
            )

        def Encoder(in_channels, out_channels, kernel_size, padding):
            return LinkConv(in_channels=in_channels, middle_channels=out_channels, out_channels=out_channels,
                            kernel_size=kernel_size, padding=padding)
        self.conv1 = Encoder(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = Encoder(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = Encoder(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv4 = Encoder(in_channels=256, out_channels=512, kernel_size=3, padding=1)

        self.conv5 = LinkConv(in_channels=512, middle_channels=1024, out_channels=512, kernel_size=3, padding=1)

        # 上采样
        def Decoder(in_channels, out_channels, kernel_size, padding):
            return LinkConv(in_channels=2 * in_channels, middle_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size, padding=padding)

        self.conv6 = Decoder(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = Decoder(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.conv8 = Decoder(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.conv9 = Decoder(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv0 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        '''
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)
        '''

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)



    def forward(self,x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.conv5(pool4)

        up1 = self.up1(conv5)
        cat1 = torch.cat((up1, conv4), dim=1)
        conv6 = self.conv6(cat1)
        up2 = self.up2(conv6)
        cat2 = torch.cat((up2, conv3), dim=1)
        conv7 = self.conv7(cat2)
        up3 = self.up3(conv7)
        cat3 = torch.cat((up3, conv2), dim=1)
        conv8 = self.conv8(cat3)
        up4 = self.up4(conv8)
        cat4 = torch.cat((up4, conv1), dim=1)
        conv9 = self.conv9(cat4)

        x = self.conv0(conv9)
        return x