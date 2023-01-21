import torch
import torch.nn as nn
import models.rfdn_baseline.block as B

def make_model(args, parent=False):
    model = RFDN()
    return model


class RFDN(nn.Module):
    def __init__(self, upscale=4):
        super(RFDN, self).__init__()

        self.fea_conv = B.conv_layer(3, 50, kernel_size=3)

        self.B1 = B.RFDB(50)
        self.B2 = B.RFDB(50)
        self.B3 = B.RFDB(50)
        self.B4 = B.RFDB(50)
        self.c = B.conv_block(50 * 4, 50, kernel_size=1, act_type='lrelu')

        self.LR_conv = B.conv_layer(50, 50, kernel_size=3)

        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(50, 3, upscale_factor=4)
        self.scale_idx = 0


    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
