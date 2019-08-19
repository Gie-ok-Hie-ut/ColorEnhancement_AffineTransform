import torch
from models import networks
from torch.autograd import Variable
moduleG = networks.ContextAwareModules()
moduleD = networks.ContextAwareModules()
moduleD.lowhighnet.conv1.weight = moduleG.lowhighnet.conv1.weight
D = networks.ContextDiscriminator(moduleD.lowhighnet, moduleD.midnet)
G = networks.ContextGenerator(moduleG.lowhighnet, moduleG.midnet)
img = Variable(torch.randn(1, 3, 224, 224))
res = G.forward(img)
