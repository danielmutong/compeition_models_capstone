import torch.nn as nn

import torch
import torch.nn.functional as F
import torchvision

#hard coded functions of conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):

#this function is hardcoded on line 188
def conv_layerdc():
    return nn.Conv2d(50, 25, 1, 1, padding=0, bias = True, dilation = 1, groups = 1)

def conv_layerrc():
    return nn.Conv2d(50, 50, 3, 1, padding = 1, bias = True, dilation = 1, groups = 1)

def conv_layerc4():
    return nn.Conv2d(50, 25, 3, 1, padding = 1, bias = True, dilation = 1, groups = 1)

def conv_layer_fea_conv():
    return nn.Conv2d(3, 50, 3, 1, padding = 1, bias = True, dilation = 1, groups = 1)

def conv_layer_LR_conv():
    return nn.Conv2d(50, 50, 3, 1, padding = 1, bias = True, dilation = 1, groups = 1)

def conv_layerc5():
    return nn.Conv2d(100, 50, 1, 1, padding = 0, bias = True, dilation = 1, groups = 1)


def conv_layer_p():
    return nn.Conv2d(50, 48, 3, 1, padding = 1, bias = True, dilation = 1, groups = 1)



def conv_block():
    act_type='lrelu'
    pad_type='zero'
    norm_type=None
    padding = get_valid_padding(1, 1)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(200, 50, 1, 1, padding = 0, dilation = 1, bias = True, groups = 1)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, 50) if norm_type else None
    return sequential(p, c, n, a)

#    def __init__(self, conv):
#        super(ESA, self).__init__()
#        f = 50 // 4
#        self.conv1 = conv(50, f, 1)
#        self.conv_f = conv(f, f, 1)
#        self.conv_max = conv(f, f, 3, 1)
#        self.conv2 = conv(f, f, 3, 2, 0)
#        self.conv3 = conv(f, f, 3, 1)
#        self.conv3_ = conv(f, f, 3, 1)
#        self.conv4 = conv(f, 50, 1)
#        self.sigmoid = nn.Sigmoid()
#        self.relu = nn.ReLU(inplace=True)


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (744 * 1296)

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (744 * 1296)
    return F_variance.pow(0.5)

def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

class ESA(nn.Module):
    def __init__(self):
        super(ESA, self).__init__()
        conv = nn.Conv2d
        f = 50 // 4
        self.conv1 = conv(50, 12, 1)
        self.conv_f = conv(12, 12, 1)
        self.conv_max = conv(12, 12, 3, 1)
        self.conv2 = conv(12, 12, 3, 2, 0)
        self.conv3 = conv(12, 12, 3, 1)
        self.conv3_ = conv(12, 12, 3, 1)
        self.conv4 = conv(12, 50, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)

        c3 = F.interpolate(c3, (744, 1296), mode='bilinear', align_corners=False) 

        cf = self.conv_f(c1_)
        c4 = self.conv4(c3+cf)
        #c4 = self.conv4(cf)
        m = self.sigmoid(c4)
        
        return x * m


class RFDB(nn.Module):
    def __init__(self, distillation_rate=0.25):
        super(RFDB, self).__init__()
        in_channels = 50
        self.dc = self.distilled_channels = in_channels//2
        self.rc = self.remaining_channels = in_channels

        #implemented a hardcoded conv_layer, changing this back to orignal conv_layer function will run with no problems
        self.c1_d = conv_layerdc()
        
        self.c1_r = conv_layerrc()
        self.c2_d = conv_layerdc()
        self.c2_r = conv_layerrc()
        self.c3_d = conv_layerdc()
        self.c3_r = conv_layerrc()
        self.c4 = conv_layerc4()
        self.act = nn.LeakyReLU(0.05, True)
        
        self.c5 = conv_layerc5()
        self.esa = ESA()

    def forward(self, input):

        distilled_c1 = self.act(self.c1_d(input))

        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1+input)
        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2+r_c1)
        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3+r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.esa(self.c5(out)) 

        return out_fused



def pixelshuffle_block():
    conv = conv_layer_p()
    pixel_shuffle = nn.PixelShuffle(4)
    return sequential(conv, 4)