import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, ngf, which_model_netG, module, norm='batch', use_dropout=False, gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'affine_resnet':
        netG = AffineGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'context_aware':
        netG = ContextGenerator(module.lowhighnet, module.midnet)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(device=gpu_ids[0])
    # print(netG)
    if which_model_netG != 'context_aware':
        netG.apply(weights_init)

    if which_model_netG == 'affine_resnet':
        last_key = list(netG.resnet.model._modules)[-1]
        netG.resnet.model._modules.get(last_key).bias.data = torch.cuda.FloatTensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])
    elif which_model_netG == 'context_aware':
        last_key = list(netG.upsampler._modules)[-1]
        netG.upsampler._modules.get(last_key).bias.data = torch.cuda.FloatTensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])
    return netG


def define_D(input_nc, ndf, which_model_netD, module, n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'context_aware':
        netD = ContextDiscriminator(module.lowhighnet, module.midnet)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda(device=gpu_ids[0])

    if which_model_netD != 'context_aware':
        netD.apply(weights_init)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

class AffineGenerator(torch.nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, gpu_ids=[], padding_type='reflect'):
        super(AffineGenerator, self).__init__()
        self.resnet = ResnetGenerator(input_nc, 12, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks, gpu_ids=gpu_ids)

    def forward(self, input):
        self.transform = self.resnet.forward(input)
        L = torch.sum(input * self.transform.narrow(1, 0, 3) , 1, keepdim=True) + self.transform.narrow(1, 3, 1)
        a = torch.sum(input * self.transform.narrow(1, 4, 3) , 1, keepdim=True) + self.transform.narrow(1, 7, 1)
        b = torch.sum(input * self.transform.narrow(1, 8, 3) , 1, keepdim=True) + self.transform.narrow(1, 11, 1)
        Lab = torch.cat((L, a, b), 1)
        return Lab

        
class ProjectiveGenerator(torch.nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, gpu_ids=[], padding_type='reflect'):
        super(ProjectiveGenerator, self).__init__()
        self.resnet = ResnetGenerator(input_nc, 12, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks, gpu_ids=gpu_ids)

    def forward(self, input):
        self.transform = self.resnet.forward(input)
        L = torch.sum(input * self.transform.narrow(1, 0, 3) , 1, keepdim=True) + self.transform.narrow(1, 3, 1)
        a = torch.sum(input * self.transform.narrow(1, 4, 3) , 1, keepdim=True) + self.transform.narrow(1, 7, 1)
        b = torch.sum(input * self.transform.narrow(1, 8, 3) , 1, keepdim=True) + self.transform.narrow(1, 11, 1)
        Lab = torch.cat((L, a, b), 1)
        return Lab

class ContextGenerator(torch.nn.Module):
    def __init__(self, lowhighnet, midnet, norm_layer=nn.BatchNorm2d, use_dropout=False, padding_type='reflect'):
        super(ContextGenerator, self).__init__()
        self.lowhighnet = lowhighnet
        self.midnet = midnet
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = []
        model += [ResnetBlock(128+512, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        model += [ResnetBlock(128+512, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        model += [nn.ConvTranspose2d(128+512, 256, kernel_size=4, stride=2, padding=1, output_padding=0, bias=use_bias),
                  norm_layer(256),
                  nn.ReLU(True)]
        model += [nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=0, bias=use_bias),
                  norm_layer(128),
                  nn.ReLU(True)]
        model += [nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0, bias=use_bias),
                  norm_layer(64),
                  nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(64, 12, kernel_size=7, padding=0)]

        self.upsampler = nn.Sequential(*model)
        self.upsampler.apply(weights_init)

    def forward(self, input):
        low, high = self.lowhighnet.forward(input)
        mid = self.midnet.forward(low)
        feature = torch.cat((mid, high.expand(high.size(0), 512, mid.size(2), mid.size(3))), 1)
        self.transform = self.upsampler.forward(feature)
        L = torch.sum(input * self.transform.narrow(1, 0, 3) , 1, keepdim=True) + self.transform.narrow(1, 3, 1)
        a = torch.sum(input * self.transform.narrow(1, 4, 3) , 1, keepdim=True) + self.transform.narrow(1, 7, 1)
        b = torch.sum(input * self.transform.narrow(1, 8, 3) , 1, keepdim=True) + self.transform.narrow(1, 11, 1)
        Lab = torch.cat((L, a, b), 1)
        return Lab

class ContextDiscriminator(torch.nn.Module):
    def __init__(self, lowhighnet, midnet, norm_layer=nn.BatchNorm2d, use_dropout=False, padding_type='reflect'):
        super(ContextDiscriminator, self).__init__()
        self.lowhighnet = lowhighnet
        self.midnet = midnet
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = []
        # model += [ResnetBlock(128+512, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        model += [nn.Conv2d(128+512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),
                 norm_layer(512),
                 nn.LeakyReLU(0.2, True)]
        model += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),
                 norm_layer(512),
                 nn.LeakyReLU(0.2, True)]
        model += [nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1, bias=use_bias),]

        self.predictor = nn.Sequential(*model)
        self.predictor.apply(weights_init)

    def forward(self, input):
        low, high = self.lowhighnet.forward(input)
        mid = self.midnet.forward(low)
        feature = torch.cat((mid, high.expand(high.size(0), 512, mid.size(2), mid.size(3))), 1)
        predict = self.predictor.forward(feature)
        return predict


class ContextAwareModules():
    def __init__(self, norm_layer=nn.BatchNorm2d, use_dropout=False, padding_type='reflect'):
        super(ContextAwareModules, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        from functools import partial
        import pickle
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

        from .wideresnet import ResLowHigh, BasicBlock
        model_file = 'models/whole_resnet18_places365.pth.tar'
        self.lowhighnet = ResLowHigh(BasicBlock, [2, 2, 2, 2], num_classes=365)
        self.lowhighnet.load_state_dict(torch.load(model_file).state_dict())
        for param in self.lowhighnet.parameters():
            param.requires_grad = False

        model = []
        model += [ResnetBlock(128, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        model += [ResnetBlock(128, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        model += [ResnetBlock(128, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        model += [ResnetBlock(128, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.midnet = nn.Sequential(*model)
        self.midnet.apply(weights_init)

# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        model += [nn.ReflectionPad2d((1, 0, 1, 0)),
                  nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=0, bias=use_bias),
                  norm_layer(ngf*2),
                  nn.ReLU(True)]
        model += [nn.ReflectionPad2d((1, 0, 1, 0)),
                  nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=0, bias=use_bias),
                  norm_layer(ngf*4),
                  nn.ReLU(True)]

        for i in range(n_blocks):
            model += [ResnetBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        model += [nn.ConvTranspose2d(ngf * 4, ngf*2, kernel_size=4, stride=2, padding=1, output_padding=0, bias=use_bias),
                  norm_layer(ngf * 2),
                  nn.ReLU(True)]
        model += [nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, output_padding=0, bias=use_bias),
                  norm_layer(ngf),
                  nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, 12, kernel_size=7, padding=0)]
        # model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class FeatureExtractor(nn.Module):
    def __init__(self, input_nc, ngf, use_bias):
        super(FeatureExtractor, self).__init__()
        net = [nn.Conv2d(input_nc, ngf, kernel_size=3, stride=1, padding=1, bias=use_bias),
               nn.PReLU(),
               nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        self.model = nn.Sequential(*net)

    def forward(self, input):
        return self.model(input)

class Mapping(nn.Module):
    def __init__(self, ngf, use_bias):
        super(Mapping, self).__init__()
        net = [nn.PReLU(),
               nn.Conv2d(ngf, 12, kernel_size=1, stride=1, padding=0, bias=use_bias),
               nn.PReLU(),
               nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1, bias=use_bias),
               nn.PReLU(),
               nn.Conv2d(12, ngf, kernel_size=1, stride=1, padding=0, bias=use_bias)]
        self.model = nn.Sequential(*net)

    def forward(self, input):
        return self.model(input)

class CNPBlock(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, submodule=None, outermost=False, innermost=False,
                norm_layer=nn.BatchNorm2d):
        super(CNPBlock, self).__init__()

        self.outermost = outermost
        self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        down = [nn.MaxPool2d(2)]
        up = [nn.ConvTranspose2d(ngf, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)]
        self.fe = FeatureExtractor(input_nc, ngf, use_bias)
        next_level = down + [submodule] + up
        self.next_level = nn.Sequential(*next_level)
        self.cur_level = Mapping(ngf, use_bias)

        if self.outermost:
            adj = [nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1, bias=use_bias),
                        nn.PReLU(),
                        nn.Conv2d(ngf, output_nc, kernel_size=1, stride=1, padding=0, bias=use_bias)]
            self.adj = nn.Sequential(*adj)

    def forward(self, input):
        feature = self.fe(input)
        if self.innermost:
            return self.cur_level(feature)
        else:
            if self.outermost:
                return self.adj(self.cur_level(feature) + self.next_level(feature))
            else:
                return self.cur_level(feature) + self.next_level(feature)

class CNPGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs=2, ngf=56, norm_layer=nn.BatchNorm2d, gpu_ids=[]):
        super(CNPGenerator, self).__init__()
        cnpblock = CNPBlock(ngf, ngf, ngf, innermost=True)
        for i in range(0, num_downs-1):
            cnpblock = CNPBlock(ngf, ngf, ngf, submodule=cnpblock)
        cnpblock = CNPBlock(input_nc, output_nc, ngf, submodule=cnpblock, outermost=True)

        self.model = cnpblock

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

# class CNPGenerator_Simple(nn.Module):
#     def __init__(self, input_nc, output_nc, num_downs=2, ngf=56, norm_layer=nn.BatchNorm2d, gpu_ids=[]):
#         super(CNPGenerator_Simple, self).__init__()
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#         self.extractors = list()
#         self.mappers = list()
#         self.extractors.append(FeatureExtractor(input_nc, ngf, use_bias))
#         self.mappers.append(Mapping(ngf, use_bias))
#         self.downers.append(nn.MaxPool2d(2))
#         self.uppers.append(nn.ConvTranspose2d(ngf, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias))

#         for i in range(0, num_downs-1):
#             self.extractors.append(FeatureExtractor(ngf, ngf, use_bias))
#             self.mappers.append(Mapping(ngf, use_bias))
#             self.downers.append(nn.MaxPool2d(2))
#             self.uppers.append(nn.ConvTranspose2d(ngf, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias))



#     def forward(self, input):
#         f[0] = self.extractors[0](input)
#         for i in range(1, N+1):
#             f[i] = self.mapper[i](self.extractors[i](self.downers[i](f[i-1])))
#         g[0] = self.mappers[0](f[0])
#         for i in range(N-1, 1, -1):
#             g[i] = torch.sum(f[i])

# class CNPGenerator(nn.Module):
#     def __init__(self, input_nc, output_nc, num_downs, ngf=56, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
#         super(CNPGenerator, self).__init__()
#         self.gpu_ids = gpu_ids
#         adj_block = [nn.Conv2d(ngf, ngf),
#                      nn.PReLU(),
#                      nn.Conv2d(ngf, output_nc)]
#         self.model = nn.Sequential(*adj_block)

#     def forward(self, input):
#         if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
#             return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
#         else:
#             return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf, ndf * 2,
                      kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [
            nn.Conv2d(ndf * 2, ndf * 4,
                      kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ndf * 4),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [
            nn.Conv2d(ndf * 4, ndf * 8,
                      kernel_size=4, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf * 8),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * 8, 1,
                      kernel_size=4, stride=1, padding=1, bias=use_bias),
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
