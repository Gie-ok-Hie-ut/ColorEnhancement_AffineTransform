import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
from pdb import set_trace as st
import random
from skimage import io, color
import numpy


class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform_gray = get_transform(opt)
        # opt.grayscale = False
        # self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        index_A = index % self.A_size
        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))

        A_img_raw = Image.open(A_path).convert('RGB')
        B_img_raw = Image.open(B_path).convert('RGB')
        # A_img = to_Lab(io.imread(A_path)) # for Lab color space
        # B_img = to_Lab(io.imread(B_path))

        A_img = self.transform_gray(A_img_raw)
        # A_aux = self.transform(A_img_raw)
        # B_img = self.transform(B_img_raw)
        B_img = self.transform_gray(B_img_raw)
        A_aux = A_img[0:3]
        B_aux = B_img[3:6]
        A_img = A_img[3:6]
        B_img = B_img[0:3]
        # print(A_img.size())
        # print(B_img.size())

        return {'A': A_img, 'B': B_img, 'A_aux': A_aux, 'B_aux': B_aux,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
