import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy
from skimage import color

def to_Lab(I):
    # AB 98.2330538631 -86.1830297444 94.4781222765 -107.857300207
    lab = color.rgb2lab(I)
    l = (lab[:, :, 0] / 100.0) * 255.0    # L component ranges from 0 to 100
    a = (lab[:, :, 1] + 86.1830297444) / (98.2330538631 + 86.1830297444) * 255.0         # a component ranges from -127 to 127
    b = (lab[:, :, 2] + 107.857300207) / (94.4781222765 + 107.857300207) * 255.0         # b component ranges from -127 to 127
    return numpy.dstack([l, a, b])

def to_Gray(I):
    # AB 98.2330538631 -86.1830297444 94.4781222765 -107.857300207
    gray = color.rgb2gray(I) * 255
    # l = (lab[:, :, 0] / 100.0) * 255.0    # L component ranges from 0 to 100
    # a = (lab[:, :, 1] + 86.1830297444) / (98.2330538631 + 86.1830297444) * 255.0         # a component ranges from -127 to 127
    # b = (lab[:, :, 2] + 107.857300207) / (94.4781222765 + 107.857300207) * 255.0         # b component ranges from -127 to 127
    return numpy.dstack([gray, gray, gray])

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
        # transform_list.append(transforms.CenterCrop(opt.fineSize))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    # transform_list.append(transforms.Lambda(                        # for Lab color space
    #         lambda img: __Lab2Tensor(img)))
    # transform_list.append(transforms.Normalize((50.0, 0.0, 0.0),    # for Lab color space
    #                                            (50.0, 100.0, 100.0)))
    if not opt.grayscale:
        # transform_list.append(transforms.Lambda(lambda img: to_Lab(numpy.array(img))))
        transform_list.append(transforms.Lambda(lambda img: (numpy.array(img))))
    else:
        transform_list.append(transforms.Lambda(lambda img: numpy.dstack([numpy.array(img), to_Gray(numpy.array(img))])))

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5, 0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __Lab2Tensor(img):
    im = torch.from_numpy(img.transpose((2,0,1)))
    return im

def __scale_width(img, target_width):
    ow, oh = img.size
    if ow > oh:
        if (oh == target_width):
            return img
        h = target_width
        w = int(target_width * ow / oh)
        return img.resize((w, h), Image.BICUBIC)
    else:
        if (ow == target_width):
            return img
        w = target_width
        h = int(target_width * oh / ow)
        return img.resize((w, h), Image.BICUBIC)
