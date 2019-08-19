import math
import numpy as np
import torch
from torch import nn
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import sys
from PIL import Image
from torchvision import transforms
sys.path.append('..')
from models import networks
from skimage import color

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

class BLTSmoothnessLoss(torch.nn.Module):
    def __init__(self, height, width, sigma_s = 1, sigma_r = 0.1):
        super(BLTSmoothnessLoss, self).__init__()
        self.sigma_s = sigma_s
        self.sigma_r = sigma_r
        self.fr = math.ceil(3*sigma_s)
        self.k_size = 2*self.fr+1
        self.pad = torch.nn.ReflectionPad2d(self.fr)
        self.height = height
        self.width = width
        self.gaussian_kernel = matlab_style_gauss2D(shape=(self.k_size, self.k_size), sigma=sigma_s)

    def computeWeightMap(self, input):
        padded_input = self.pad(input).data
        widx = 0
        self.weight_map = torch.FloatTensor(self.k_size**2, input.size(0), 1, input.size(2), input.size(3)).zero_().cuda(4)
        self.x = (input**2).sum(1, keepdim=True)
        for k in range(0, self.k_size): # height
            for l in range(0, self.k_size): # width
                # if k == self.fr and l == self.fr:
                #     widx += 1
                #     continue
                shifted_input = padded_input.narrow(2, k, self.height).narrow(3, l, self.width)
                self.weight_map[widx] = (-((input - shifted_input)**2).sum(1, keepdim=True)/(2*self.sigma_r*self.sigma_r)).exp() * self.gaussian_kernel[k][l]
                widx += 1

    def do_filter(self, input):
        padded_input = self.pad(input)
        widx = 0
        weight_map = Variable(self.weight_map)
        self.weight_sum = Variable(torch.FloatTensor(weight_map[0].data.size()).cuda(4).zero_())
        for k in range(0, self.k_size): # height
            for l in range(0, self.k_size): # width
                shifted_input = padded_input.narrow(2, k, self.height).narrow(3, l, self.width)
                # loss_map += (input - shifted_input).abs() * weight_map[widx]
                if k == 0 and l == 0:
                    self.res = shifted_input * weight_map[widx].repeat(1, shifted_input.size(1), 1, 1)
                    self.weight_sum += weight_map[widx]
                else:
                    self.res += shifted_input * weight_map[widx].repeat(1, shifted_input.size(1), 1, 1)
                    self.weight_sum += weight_map[widx]
                widx += 1
        return self.res / self.weight_sum.repeat(1, self.res.size(1), 1, 1)

    def forward(self, input):
        padded_input = self.pad(input)
        widx = 0
        weight_map = Variable(self.weight_map)
        for k in range(0, self.k_size): # height
            for l in range(0, self.k_size): # width
                shifted_input = padded_input.narrow(2, k, self.height).narrow(3, l, self.width)
                wmap = weight_map[widx].repeat(1, input.size(1), 1, 1)
                if k == 0 and l == 0:
                    self.loss_map = (input - shifted_input).abs() * wmap
                else:
                    self.loss_map += (input - shifted_input).abs() * wmap
                widx += 1
        return self.loss_map.mean()


def to_Lab(I):
    # AB 98.2330538631 -86.1830297444 94.4781222765 -107.857300207
    lab = color.rgb2lab(I)
    l = (lab[:, :, 0] / 100.0) * 255.0    # L component ranges from 0 to 100
    a = (lab[:, :, 1] + 86.1830297444) / (98.2330538631 + 86.1830297444) * 255.0         # a component ranges from -127 to 127
    b = (lab[:, :, 2] + 107.857300207) / (94.4781222765 + 107.857300207) * 255.0         # b component ranges from -127 to 127
    return np.dstack([l, a, b]).astype(np.uint8)

def to_RGB(I):
    # print(I)
    l = I[:, :, 0] / 255.0 * 100.0
    a = I[:, :, 1] / 255.0 * (98.2330538631 + 86.1830297444) - 86.1830297444
    b = I[:, :, 2] / 255.0 * (94.4781222765 + 107.857300207) - 107.857300207
    # print(np.dstack([l, a, b]))
    rgb = color.lab2rgb(np.dstack([l, a, b]).astype(np.float64))*255
    return rgb

def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    # xx = Image.fromarray(image_numpy.astype(imtype), mode='YCbCr')
    # xx = Image.fromarray(image_numpy.astype(imtype), mode='HSV')
    # image_numpy = np.array((xx.convert('RGB')))
    image_numpy = to_RGB(image_numpy)
    return image_numpy.astype(imtype)

img = Image.open('tmp.jpg').convert('RGB')
width, height = img.size

transform_list = [transforms.CenterCrop(400)]
width =400
height=400
transform_list.append(transforms.Lambda(
          lambda img: to_Lab(np.array(img))))
transform_list += [transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5),
                                        (0.5, 0.5, 0.5))]
trans = transforms.Compose(transform_list)

# trans = transforms.Compose([transforms.CenterCrop(400), transforms.ToTensor(),])
input = trans(img).view(1, 3, height, width).cuda(4)

# layer.computeWeightMap(input)
# input_v = Variable(input)
# res = layer.do_filter(input_v)

# out_trans =  transforms.Compose([transforms.ToPILImage()])
# res_img = out_trans(res.data[0].cpu())
# res_img.save('img_filtered.jpg')
input_v = Variable(input)

netG = networks.define_G(3,3,32,'affine_resnet_9blocks',norm='batch',use_dropout=True,gpu_ids=[4])
netG.load_state_dict(torch.load('40_net_G_A.pth'))
res = netG.forward(input_v)

# input_np = tensor2im(input_v.data)
# Image.fromarray(input_np).save('img_cropped.jpg')

# res_np = tensor2im(res.data)
# Image.fromarray(res_np).save('img_enhanced.jpg')

# # tmap = np.transpose(netG.transform.data[0][0:3].view(3, height, width).cpu().numpy()*255, (1,2,0))
# # Image.fromarray(tmap.astype(np.uint8), mode='RGB').save('mapL.jpg')
# tmap = ((netG.transform.data[0][0].view(height, width).cpu()+1)/2.0).numpy()*255
# Image.fromarray(tmap.astype(np.uint8), mode='L').save('mapL_1.jpg')
# tmap = ((netG.transform.data[0][1].view(height, width).cpu()+1)/2.0).numpy()*255
# Image.fromarray(tmap.astype(np.uint8), mode='L').save('mapL_2.jpg')
# tmap = ((netG.transform.data[0][2].view(height, width).cpu()+1)/2.0).numpy()*255
# Image.fromarray(tmap.astype(np.uint8), mode='L').save('mapL_3.jpg')
# tmap = ((netG.transform.data[0][3].view(height, width).cpu()+1)/2.0).numpy()*255
# Image.fromarray(tmap.astype(np.uint8), mode='L').save('mapL_4.jpg')


# tmap = ((netG.transform.data[0][4].view(height, width).cpu()+1)/2.0).numpy()*255
# Image.fromarray(tmap.astype(np.uint8), mode='L').save('mapa_1.jpg')
# tmap = ((netG.transform.data[0][5].view(height, width).cpu()+1)/2.0).numpy()*255
# Image.fromarray(tmap.astype(np.uint8), mode='L').save('mapa_2.jpg')
# tmap = ((netG.transform.data[0][6].view(height, width).cpu()+1)/2.0).numpy()*255
# Image.fromarray(tmap.astype(np.uint8), mode='L').save('mapa_3.jpg')
# tmap = ((netG.transform.data[0][7].view(height, width).cpu()+1)/2.0).numpy()*255
# Image.fromarray(tmap.astype(np.uint8), mode='L').save('mapa_4.jpg')


# tmap = ((netG.transform.data[0][8].view(height, width).cpu()+1)/2.0).numpy()*255
# Image.fromarray(tmap.astype(np.uint8), mode='L').save('mapb_1.jpg')
# tmap = ((netG.transform.data[0][9].view(height, width).cpu()+1)/2.0).numpy()*255
# Image.fromarray(tmap.astype(np.uint8), mode='L').save('mapb_2.jpg')
# tmap = ((netG.transform.data[0][10].view(height, width).cpu()+1)/2.0).numpy()*255
# Image.fromarray(tmap.astype(np.uint8), mode='L').save('mapb_3.jpg')
# tmap = ((netG.transform.data[0][11].view(height, width).cpu()+1)/2.0).numpy()*255
# Image.fromarray(tmap.astype(np.uint8), mode='L').save('mapb_4.jpg')

# tmap = np.transpose(netG.transform.data[0][4:7].view(3, height, width).cpu().numpy()*255, (1,2,0))
# Image.fromarray(tmap.astype(np.uint8), mode='RGB').save('mapA.jpg')
# tmap = netG.transform.data[0][7].view(height, width).cpu().numpy()*255
# Image.fromarray(tmap.astype(np.uint8), mode='L').save('mapA_b.jpg')

# tmap = np.transpose(netG.transform.data[0][8:11].view(3, height, width).cpu().numpy()*255, (1,2,0))
# Image.fromarray(tmap.astype(np.uint8), mode='RGB').save('mapB.jpg')
# tmap = netG.transform.data[0][11].view(height, width).cpu().numpy()*255
# Image.fromarray(tmap.astype(np.uint8), mode='L').save('mapB_b.jpg')


layer = BLTSmoothnessLoss(400, 400, sigma_s=1.5)
layer.computeWeightMap(input)
loss = layer.forward(netG.transform)
Image.fromarray((layer.loss_map.data[0].mean(0).cpu().numpy()*10*255).astype(np.uint8), mode='L').save('loss.jpg')