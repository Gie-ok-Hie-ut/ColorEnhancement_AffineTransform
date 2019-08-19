import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys
import math

# def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
#     """
#     2D gaussian mask - should give the same result as MATLAB's
#     fspecial('gaussian',[shape],[sigma])
#     """
#     m,n = [(ss-1.)/2. for ss in shape]
#     y,x = np.ogrid[-m:m+1,-n:n+1]
#     h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
#     h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
#     sumh = h.sum()
#     if sumh != 0:
#         h /= sumh
#     return h

# class BLTSmoothnessLoss(torch.nn.Module):
#     def __init__(self, height, width, sigma_s = 1, sigma_r = 0.1):
#         super(BLTSmoothnessLoss, self).__init__()
#         self.sigma_s = sigma_s
#         self.sigma_r = sigma_r
#         self.fr = math.ceil(3*sigma_s)
#         self.k_size = 2*self.fr+1
#         self.pad = torch.nn.ReflectionPad2d(self.fr)
#         self.height = height
#         self.width = width
#         self.gaussian_kernel = matlab_style_gauss2D(shape=(self.k_size, self.k_size), sigma=sigma_s)

#     def computeWeightMap(self, input):
#         padded_input = self.pad(input).data
#         widx = 0
#         self.weight_map = torch.FloatTensor(self.k_size**2, input.size(0), 1, input.size(2), input.size(3)).zero_().cuda(4)
#         self.x = (input**2).sum(1, keepdim=True)
#         for k in range(0, self.k_size): # height
#             for l in range(0, self.k_size): # width
#                 # if k == self.fr and l == self.fr:
#                 #     widx += 1
#                 #     continue
#                 shifted_input = padded_input.narrow(2, k, self.height).narrow(3, l, self.width)
#                 self.weight_map[widx] = (-((input - shifted_input)**2).sum(1, keepdim=True)/(2*self.sigma_r*self.sigma_r)).exp() * self.gaussian_kernel[k][l]
#                 widx += 1

#     def do_filter(self, input):
#         padded_input = self.pad(input)
#         widx = 0
#         weight_map = Variable(self.weight_map)
#         self.weight_sum = Variable(torch.FloatTensor(weight_map[0].data.size()).cuda(4).zero_())
#         for k in range(0, self.k_size): # height
#             for l in range(0, self.k_size): # width
#                 shifted_input = padded_input.narrow(2, k, self.height).narrow(3, l, self.width)
#                 # loss_map += (input - shifted_input).abs() * weight_map[widx]
#                 if k == 0 and l == 0:
#                     self.res = shifted_input * weight_map[widx].repeat(1, shifted_input.size(1), 1, 1)
#                     self.weight_sum += weight_map[widx]
#                 else:
#                     self.res += shifted_input * weight_map[widx].repeat(1, shifted_input.size(1), 1, 1)
#                     self.weight_sum += weight_map[widx]
#                 widx += 1
#         return self.res / self.weight_sum.repeat(1, self.res.size(1), 1, 1)

#     def forward(self, input):
#         padded_input = self.pad(input)
#         widx = 0
#         weight_map = Variable(self.weight_map)
#         for k in range(0, self.k_size): # height
#             for l in range(0, self.k_size): # width
#                 shifted_input = padded_input.narrow(2, k, self.height).narrow(3, l, self.width)
#                 wmap = weight_map[widx].repeat(1, input.size(1), 1, 1)
#                 if k == 0 and l == 0:
#                     self.loss_map = (input - shifted_input).abs() * wmap
#                 else:
#                     self.loss_map += (input - shifted_input).abs() * wmap
#                 widx += 1
#         return self.loss_map.mean()


class ColorEnhanceModel(BaseModel):
    def name(self):
        return 'ColorEnhanceModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.alpha_ratio = 0.1

        nb = opt.batchSize
        size = opt.fineSize
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)
        self.input_A_aux = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B_aux = self.Tensor(nb, opt.output_nc, size, size)

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.contextModule_G_A = networks.ContextAwareModules()
        self.contextModule_D_A = networks.ContextAwareModules()
        self.contextModule_G_B = networks.ContextAwareModules()
        self.contextModule_D_B = networks.ContextAwareModules()

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, self.contextModule_G_A,
                                        opt.norm, not opt.no_dropout, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.which_model_netG, self.contextModule_G_B,
                                        opt.norm, not opt.no_dropout, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD, self.contextModule_D_A,
                                            opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD, self.contextModule_D_B,
                                            opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            print('load from epoch ', which_epoch)
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            # self.smoothnessLoss_A = BLTSmoothnessLoss(opt.fineSize, opt.fineSize, sigma_s=1.5, sigma_r=0.2)
            # self.smoothnessLoss_B = BLTSmoothnessLoss(opt.fineSize, opt.fineSize, sigma_s=1.5, sigma_r=0.2)

            # initialize optimizers
            if opt.which_model_netG != 'context_aware':
                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
            else:
                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.midnet.parameters(), self.netG_B.midnet.parameters(),
                                                                    self.netG_A.upsampler.parameters(), self.netG_B.upsampler.parameters()),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
            if opt.which_model_netD != 'context_aware':
                self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            else:
                self.optimizer_D_A = torch.optim.Adam(itertools.chain(self.netD_A.midnet.parameters(),self.netD_A.predictor.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D_B = torch.optim.Adam(itertools.chain(self.netD_B.midnet.parameters(),self.netD_B.predictor.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        networks.print_network(self.netG_B)
        if self.isTrain:
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        input_A_aux = input['A_aux' if AtoB else 'B_aux']
        input_B_aux = input['B_aux' if AtoB else 'A_aux']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.input_A_aux.resize_(input_A_aux.size()).copy_(input_A_aux)
        self.input_B_aux.resize_(input_B_aux.size()).copy_(input_B_aux)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_A_aux = Variable(self.input_A_aux)
        self.real_B_aux = Variable(self.input_B_aux)
        # self.smoothnessLoss_A.computeWeightMap(self.input_A)
        # self.smoothnessLoss_B.computeWeightMap(self.input_B)

    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG_A.forward(self.real_A)
        self.rec_A = self.netG_B.forward(self.fake_B)

        self.real_B = Variable(self.input_B, volatile=True)
        self.fake_A = self.netG_B.forward(self.real_B)
        self.rec_B = self.netG_A.forward(self.fake_A)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD.forward(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD.forward(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_A.forward(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B = self.netG_B.forward(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss
        # D_A(G_A(A))
        self.fake_B = self.alpha_ratio * self.netG_A.forward(self.real_A) + (1-self.alpha_ratio) * self.real_A_aux
        pred_fake = self.netD_A.forward(self.fake_B)
        self.loss_G_A = self.criterionGAN(pred_fake, True)
        # D_B(G_B(B))
        self.fake_A = self.alpha_ratio * self.netG_B.forward(self.real_B) + (1-self.alpha_ratio) * self.real_B_aux
        pred_fake = self.netD_B.forward(self.fake_A)
        self.loss_G_B = self.criterionGAN(pred_fake, True)
        # Forward cycle loss
        self.rec_A = self.alpha_ratio * self.netG_B.forward(self.fake_B) + (1-self.alpha_ratio) * self.real_A
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss
        self.rec_B = self.alpha_ratio * self.netG_A.forward(self.fake_A) + (1-self.alpha_ratio) * self.real_B
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # bilateral smoothness loss
        # self.blt_A = self.smoothnessLoss_A.forward(self.netG_A.transform) * 5
        # self.blt_B = self.smoothnessLoss_B.forward(self.netG_B.transform) * 5

        # combined loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B \
         + self.loss_idt_A + self.loss_idt_B \
         # + self.blt_A + self.blt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()

    def get_current_errors(self):
        D_A = self.loss_D_A.data[0]
        G_A = self.loss_G_A.data[0]
        Cyc_A = self.loss_cycle_A.data[0]
        D_B = self.loss_D_B.data[0]
        G_B = self.loss_G_B.data[0]
        Cyc_B = self.loss_cycle_B.data[0]
        if self.opt.identity > 0.0:
            idt_A = self.loss_idt_A.data[0]
            idt_B = self.loss_idt_B.data[0]
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A), ('idt_A', idt_A),
                                ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B), ('idt_B', idt_B),])
                                # ('S_A', self.blt_A.data[0]), ('S_B', self.blt_B.data[0])])
        else:
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A),
                                ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B),])
                                # ('S_A', self.blt_A.data[0]), ('S_B', self.blt_B.data[0])])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        rec_A = util.tensor2im(self.rec_A.data)
        real_B = util.tensor2im(self.real_B.data)
        fake_A = util.tensor2im(self.fake_A.data)
        rec_B = util.tensor2im(self.rec_B.data)
        if self.opt.identity > 0.0:
            idt_A = util.tensor2im(self.idt_A.data)
            idt_B = util.tensor2im(self.idt_B.data)
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A), ('idt_B', idt_B),
                                ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B), ('idt_A', idt_A)])
        else:
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                                ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B)])

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D_A.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D_B.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def update_alpha_ratio(self, alpha_ratio):
        self.alpha_ratio = alpha_ratio
