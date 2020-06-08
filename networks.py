import numpy as np
import torch 
import torch.nn as nn
import sklearn
import os
import cv2
import torchvision
import torchvision.transforms as transforms
import PIL
import tqdm
from tqdm import tqdm_notebook as tqdm
import torch.utils.model_zoo as model_zoo
import matplotlib.pyplot as plt
import torchvision.models as models
from torchsummary import summary
from torch.nn import init
import torch.nn.functional as F
from math import sqrt
import torchvision


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg16_bn' : 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
}

sizes = [38 , 19 , 10 , 5 , 3 , 1] 
ech = [0.1 , 0.2 , 0.375 , 0.55 , 0.725 , 0.9] 
dilatations = [[1., 2., 0.5],[1., 2., 3., 0.5, .333],[1., 2., 3., 0.5, .333],[1., 2., 3., 0.5, .333],[1., 2., 0.5], [1., 2., 0.5]]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def decimate(tensor, m):
    """
    Decimate a tensor by a factor 'm', i.e. downsample by keeping every 'm'th value.
    This is used when we convert FC layers to equivalent Convolutional layers, BUT of a smaller size.
    :param tensor: tensor to be decimated
    :param m: list of decimation factors for each dimension of the tensor; None if not to be decimated along a dimension
    :return: decimated tensor
    """
    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d,
                                         index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())

    return tensor
    
class VGGBase(nn.Module):
    """
    VGG base convolutions to produce lower-level feature maps.
    """

    def __init__(self):
        super(VGGBase, self).__init__()

        # Standard convolutional layers in VGG16
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # stride = 1, by default
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # ceiling (not floor) here for even dims

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # retains size because stride is 1 (and padding)

        # Replacements for FC6 and FC7 in VGG16
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)  # atrous convolution

        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

        # Load pretrained layers
        self.load_pretrained_layers()

    def forward(self, image):
        """
        Forward propagation.
        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: lower-level feature maps conv4_3 and conv7
        """
        out = F.relu(self.conv1_1(image))  # (N, 64, 300, 300)
        out = F.relu(self.conv1_2(out))  # (N, 64, 300, 300)
        out = self.pool1(out)  # (N, 64, 150, 150)

        out = F.relu(self.conv2_1(out))  # (N, 128, 150, 150)
        out = F.relu(self.conv2_2(out))  # (N, 128, 150, 150)
        out = self.pool2(out)  # (N, 128, 75, 75)

        out = F.relu(self.conv3_1(out))  # (N, 256, 75, 75)
        out = F.relu(self.conv3_2(out))  # (N, 256, 75, 75)
        out = F.relu(self.conv3_3(out))  # (N, 256, 75, 75)
        out = self.pool3(out)  # (N, 256, 38, 38), it would have been 37 if not for ceil_mode = True

        out = F.relu(self.conv4_1(out))  # (N, 512, 38, 38)
        out = F.relu(self.conv4_2(out))  # (N, 512, 38, 38)
        out = F.relu(self.conv4_3(out))  # (N, 512, 38, 38)
        conv4_3_feats = out  # (N, 512, 38, 38)
        out = self.pool4(out)  # (N, 512, 19, 19)

        out = F.relu(self.conv5_1(out))  # (N, 512, 19, 19)
        out = F.relu(self.conv5_2(out))  # (N, 512, 19, 19)
        out = F.relu(self.conv5_3(out))  # (N, 512, 19, 19)
        out = self.pool5(out)  # (N, 512, 19, 19), pool5 does not reduce dimensions

        out = F.relu(self.conv6(out))  # (N, 1024, 19, 19)

        conv7_feats = F.relu(self.conv7(out))  # (N, 1024, 19, 19)

        # Lower-level feature maps
        return conv4_3_feats, conv7_feats

    def load_pretrained_layers(self):
        """
        As in the paper, we use a VGG-16 pretrained on the ImageNet task as the base network.
        There's one available in PyTorch, see https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.vgg16
        We copy these parameters into our network. It's straightforward for conv1 to conv5.
        However, the original VGG-16 does not contain the conv6 and con7 layers.
        Therefore, we convert fc6 and fc7 into convolutional layers, and subsample by decimation. See 'decimate' in utils.py.
        """
        # Current state of base
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # Pretrained VGG base
        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())

        # Transfer conv. parameters from pretrained model to current model
        for i, param in enumerate(param_names[:-4]):  # excluding conv6 and conv7 parameters
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        # Convert fc6, fc7 to convolutional layers, and subsample (by decimation) to sizes of conv6 and conv7
        # fc6
        conv_fc6_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)  # (4096, 512, 7, 7)
        conv_fc6_bias = pretrained_state_dict['classifier.0.bias']  # (4096)
        state_dict['conv6.weight'] = decimate(conv_fc6_weight, m=[4, None, 3, 3])  # (1024, 512, 3, 3)
        state_dict['conv6.bias'] = decimate(conv_fc6_bias, m=[4])  # (1024)
        # fc7
        conv_fc7_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)  # (4096, 4096, 1, 1)
        conv_fc7_bias = pretrained_state_dict['classifier.3.bias']  # (4096)
        state_dict['conv7.weight'] = decimate(conv_fc7_weight, m=[4, 4, None, None])  # (1024, 1024, 1, 1)
        state_dict['conv7.bias'] = decimate(conv_fc7_bias, m=[4])  # (1024)

        # Note: an FC layer of size (K) operating on a flattened version (C*H*W) of a 2D image of size (C, H, W)...
        # ...is equivalent to a convolutional layer with kernel size (H, W), input channels C, output channels K...
        # ...operating on the 2D image of size (C, H, W) without padding

        self.load_state_dict(state_dict)

        print("\nLoaded base model.\n")


class VGG(nn.Module): #Réseau VGG tronqué et modifié 

    def __init__(self, features_1,features_2, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features_1 = features_1
        self.features_2 = features_2
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()
        
        self.load_pretrained_layers()

    def forward(self, x):
        y = self.features_1(x)
        x = self.features_2(y)
        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        #x = self.classifier(x)
        return y , x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def load_pretrained_layers(self):
        """
        As in the paper, we use a VGG-16 pretrained on the ImageNet task as the base network.
        There's one available in PyTorch, see https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.vgg16
        We copy these parameters into our network. It's straightforward for conv1 to conv5.
        However, the original VGG-16 does not contain the conv6 and con7 layers.
        Therefore, we convert fc6 and fc7 into convolutional layers, and subsample by decimation. See 'decimate' in utils.py.
        """
        # Current state of base
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # Pretrained VGG base
        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())

        # Transfer conv. parameters from pretrained model to current model
        for i, param in enumerate(param_names[:-4]):  # excluding conv6 and conv7 parameters
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        # Convert fc6, fc7 to convolutional layers, and subsample (by decimation) to sizes of conv6 and conv7
        # fc6
        conv_fc6_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)  # (4096, 512, 7, 7)
        conv_fc6_bias = pretrained_state_dict['classifier.0.bias']  # (4096)
        state_dict['conv6.weight'] = decimate(conv_fc6_weight, m=[4, None, 3, 3])  # (1024, 512, 3, 3)
        state_dict['conv6.bias'] = decimate(conv_fc6_bias, m=[4])  # (1024)
        # fc7
        conv_fc7_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)  # (4096, 4096, 1, 1)
        conv_fc7_bias = pretrained_state_dict['classifier.3.bias']  # (4096)
        state_dict['conv7.weight'] = decimate(conv_fc7_weight, m=[4, 4, None, None])  # (1024, 1024, 1, 1)
        state_dict['conv7.bias'] = decimate(conv_fc7_bias, m=[4])  # (1024)

        # Note: an FC layer of size (K) operating on a flattened version (C*H*W) of a 2D image of size (C, H, W)...
        # ...is equivalent to a convolutional layer with kernel size (H, W), input channels C, output channels K...
        # ...operating on the 2D image of size (C, H, W) without padding

        self.load_state_dict(state_dict)

        print("\nLoaded base model.\n")


def make_layers(cfg, depth = 3, batch_norm=False , SSD = False):
    layers = []
    param = cfg
    n = len(param)
    for i in range(n):
      if param[i] == 'M':
        layers.append(nn.MaxPool2d(2 , stride =  2 , ceil_mode= True))
        continue
      layers.append(nn.Conv2d(depth,param[i],3, padding = 1))
      depth = param[i]
      if batch_norm:
         layers.append(nn.BatchNorm2d(depth))
      layers.append(nn.ReLU())

    if SSD:
      layers.append(nn.MaxPool2d(3 , stride = 1 , padding =1))
      layers.append(nn.Conv2d(depth ,depth*2 , 3 , padding = 6 , dilation=6))
      layers.append(nn.ReLU())
      layers.append(nn.Conv2d(depth*2 , depth*2 , 1))
      layers.append(nn.ReLU())
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D' : [64 , 64 , 'M' , 128 , 128 , 'M' , 256 , 256 , 256 , 'M' , 512 , 512 , 512 , 'M' , 512 , 512 , 512],
    'SSD1' : [64 , 64 , 'M' , 128 , 128 , 'M' , 256 , 256 , 256 , 'M' , 512 , 512 , 512],
    'SSD2' : ['M' , 512 , 512 , 512]
}

def vgg_16_classifier(num_classes):
  classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )
  return classifier
  
def vgg16(num_classes, pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['SSD1'], batch_norm=True),make_layers(cfg['SSD2'],  depth = 512,batch_norm=False, SSD = True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    model.classifier = vgg_16_classifier(num_classes)
    return model

def IOUs(set_1, set_2):
        lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  
        upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  
        intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)
        intersection = intersection_dims[:, :, 0] * intersection_dims[:, :, 1]
        #print(intersection)
        areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  
        areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  
        union = (areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection)  

        return intersection / union 


class BoxRegressionNet(nn.Module):

    def __init__(self):
        super(BoxRegressionNet, self).__init__()

        self.conv1_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0)  
        self.conv1_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1) 

        self.conv2_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv2_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) 

        self.conv3_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv3_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0) 

        self.conv4_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv4_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0) 

        #Initialiser les poids 

        self.init_conv2d()

    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)
  

    def forward(self, x):

        out = F.relu(self.conv1_1(x)) 
        out = F.relu(self.conv1_2(out)) 
        fmap1 = out 

        out = F.relu(self.conv2_1(out)) 
        out = F.relu(self.conv2_2(out)) 
        fmap2 = out 

        out = F.relu(self.conv3_1(out)) 
        out = F.relu(self.conv3_2(out))  
        fmap3 = out 

        out = F.relu(self.conv4_1(out)) 
        fmap4 = F.relu(self.conv4_2(out))  

        return fmap1, fmap2, fmap3, fmap4


class ClassificationNet(nn.Module):

        def __init__(self, nbr_classes):

            super(ClassificationNet, self).__init__()

        # Nous prenons respectivement pour chaque features map le nombre de priors par "case" suivant: 4,6,6,6,4,4 

        # Couches de convolution de prediction de localisation
            self.loc_conv1 = nn.Conv2d(512 , 4 * 4, 3, 1, 1)
            self.loc_conv2 = nn.Conv2d(1024, 6 * 4, 3, 1, 1)
            self.loc_conv3 = nn.Conv2d(512, 6 * 4, 3, 1, 1)
            self.loc_conv4 = nn.Conv2d(256, 6 * 4, 3, 1, 1)
            self.loc_conv5 = nn.Conv2d(256, 4 * 4, 3, 1, 1)
            self.loc_conv6 = nn.Conv2d(256, 4 * 4, 3, 1, 1)

        # Couches de convolution de  prediction de classe
            self.cla_conv1 = nn.Conv2d(512, 4 * nbr_classes, 3, 1, 1)
            self.cla_conv2 = nn.Conv2d(1024, 6 * nbr_classes, 3, 1, 1)
            self.cla_conv3 = nn.Conv2d(512, 6 * nbr_classes, 3, 1, 1)
            self.cla_conv4 = nn.Conv2d(256, 6 * nbr_classes, 3, 1, 1)
            self.cla_conv5 = nn.Conv2d(256, 4 * nbr_classes, 3, 1, 1)
            self.cla_conv6 = nn.Conv2d(256, 4 * nbr_classes, 3, 1, 1)

            self.nbr_classes = nbr_classes

        def conv_and_organize(self ,features, conv, output_width):
            batch_size = features.size(0)
            #print(batch_size)
            res = conv(features)
            res = res.permute(0, 2, 3, 1).contiguous()
            res = res.view(batch_size, -1, output_width)
            return res
    
        def forward(self, features1, features2, features3, features4, features5, features6):
        
        # Prédiction de localisation : size=(batch_size, nbr_prior, 4)
            print(features1.size())
            loc1 = self.conv_and_organize(features1, self.loc_conv1, 4)
            loc2 = self.conv_and_organize(features2, self.loc_conv2, 4)
            loc3 = self.conv_and_organize(features3, self.loc_conv3, 4)
            loc4 = self.conv_and_organize(features4, self.loc_conv4, 4)
            loc5 = self.conv_and_organize(features5, self.loc_conv5, 4)
            loc6 = self.conv_and_organize(features6, self.loc_conv5, 4)

        # Prédiction de classe : size=(batch_size, nbr_prior, nbr_classes)
            cla1 = self.conv_and_organize(features1, self.cla_conv1, self.nbr_classes)
            cla2 = self.conv_and_organize(features2, self.cla_conv2, self.nbr_classes)
            cla3 = self.conv_and_organize(features3, self.cla_conv3, self.nbr_classes)
            cla4 = self.conv_and_organize(features4, self.cla_conv4, self.nbr_classes)
            cla5 = self.conv_and_organize(features5, self.cla_conv5, self.nbr_classes)
            cla6 = self.conv_and_organize(features6, self.cla_conv6, self.nbr_classes)

            loc = torch.cat([loc1, loc2, loc3, loc4, loc5, loc6], dim=1)  
            cla = torch.cat([cla1, cla2, cla3, cla4, cla5, cla6], dim=1)  

            return loc, cla    


def center(rect):
    s= rect.size()[0]
    x_min , y_min , w , h = rect[:,0].view((s,1)) , rect[:,1].view((s,1)), rect[:,2].view((s,1)) ,  rect[:,3].view((s,1))
    return torch.cat([(x_min + w) / 2, (y_min + h)/2, w , h ], 1) 


def uncenter(rect):
    s= rect.size()[0]
    print(rect.size())
    x , y , w , h = rect[:,0].view((s,1)) , rect[:,1].view((s,1)) , rect[:,2].view((s,1)) , rect[:,3].view((s,1))
    print(x.size())
    return torch.cat([x - w / 2, y - h /2, w, h], dim = 1) 
      

def deviate(rect , default_boxes):
    s= rect.size()[0]
    x , y , w , h = rect[:,0].view((s,1)) , rect[:,1].view((s,1)) , rect[:,2].view((s,1)) , rect[:,3].view((s,1))
    x_d , y_d , w_d , h_d = default_boxes[:,0].view((s,1)) , default_boxes[:,1].view((s,1)) , default_boxes[:,2].view((s,1)) , default_boxes[:,3].view((s,1))
    return torch.cat([(x - x_d) / w_d , (y - y_d )/ h_d , torch.log(w / w_d ) , torch.log(h/h_d)], 1)  


def undeviate(rect , default_boxes):
    s= rect.size()[0]
    x , y , w , h = rect[:,0].view((s,1)) , rect[:,1].view((s,1)) , rect[:,2].view((s,1)) , rect[:,3].view((s,1))
    x_d , y_d , w_d , h_d = default_boxes[:,0].view((s,1)) , default_boxes[:,1].view((s,1)) , default_boxes[:,2].view((s,1)) , default_boxes[:,3].view((s,1))
    return torch.cat([x * w_d  + x_d,y*h_d + y_d ,  torch.exp(w) * w_d , torch.exp(h)*h_d], 1)  



def default_boxes(dilatations = dilatations , sizes = sizes , ech = ech):
        default_boxes = []
        for k in range(6):
            for i in range(sizes[k]):
                x = (i+0.5) / sizes[k]
                #print(x)
                for j in range(sizes[k]): 
                    y = (j+0.5) / sizes[k]
                    for d in dilatations[k]:
                        default_boxes.append([x, y, ech[k] * sqrt(d), ech[k] / sqrt(d)])
                        if d == 1.:
                            if k<5:
                                default_boxes.append([x, y, sqrt(ech[k] * ech[k + 1]), sqrt(ech[k] * ech[k + 1])])
                            else:
                                default_boxes.append([x, y, 1, 1])

        default_boxes = torch.Tensor(default_boxes).float().cuda()
        default_boxes.clamp_(0, 1)
        print(default_boxes.size())
        return default_boxes

    

class ObjectDetection_SSD(nn.Module):
  def __init__(self, nbr_classes = 1000): #Initialisation du système SSD
    
    super(ObjectDetection_SSD, self).__init__() 

    self.cnn = VGGBase()  #Base du réseau VGG16, sans la partie Dense, renvoie la sortie de 2 layers
    self.box = BoxRegressionNet()  #Réseau de génération des rectangles (Regression)
    self.pred = ClassificationNet(nbr_classes) #Réseau de classification, renvoie les localisations des rectangles et les prédictions pour les nbr_classes classes pour chacun d'eux
    self.default_boxes = default_boxes()
    self.nbr_classes = nbr_classes
  def forward(self , x):

    feature_map1 , feature_map2 = self.cnn(x)
    feature_map3 , feature_map4, feature_map5, feature_map6 = self.box(feature_map2) 
    boxes , scores = self.pred(feature_map1 , feature_map2 ,feature_map3 ,feature_map4 ,feature_map5 ,feature_map6)

    return boxes , scores

  def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Decipher the 8732 locations and class scores (output of ths SSD300) to detect objects.
        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.
        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.default_boxes.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            print('batch_img' , i+1)
            # Decode object coordinates from the form we regressed predicted boxes to
            decoded_locs = uncenter(
                undeviate(predicted_locs[i], self.default_boxes))  # (8732, 4), these are fractional pt. coordinates

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)

            # Check for each class
            for c in range(1, self.nbr_classes):
                print('class' , c+1)
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (8732)
                score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                # Find the overlap between predicted boxes
                overlap = IOUs(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                # Non-Maximum Suppression (NMS)

                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  # (n_qualified)

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue

                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    cond = (overlap[box] > max_overlap).byte()
                    suppress = torch.max(suppress, cond )
                    # The max operation retains previously suppressed boxes, like an 'OR' operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = 0

                # Store only unsuppressed boxes for this class
                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[1 - suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size

 




class LossFunction(nn.Module):
    
    def __init__(self, default_boxes, threshold=0.5, ratio =3, alpha=1.):
        
        super(LossFunction, self).__init__()
        
        
        self.threshold = threshold
        self.ratio = ratio
        self.alpha = alpha
        self.default_boxes = default_boxes
        self.uncentered_default = uncenter(default_boxes)

        self.L1Loss = nn.L1Loss()
        self.CELoss = nn.CrossEntropyLoss(reduce=False)
        

    def forward(self, predicted_boxes, predicted_labels, boxes, labels):      
        #boxes = list(boxes)
        #labels = list(labels)
        batch_size = predicted_boxes.size(0)
        nbr_default = self.default_boxes.size(0)
        #print('predicted_scores' , predicted_scores.size())
        ground_truth_boxes = torch.zeros((batch_size, nbr_default, 4)).float().cuda()  
        ground_truth_classes = torch.zeros((batch_size, nbr_default)).cuda()
        #print(ground_truth_classes.size())
        for i in range(batch_size):
            if len(labels[i])==0:
                boxes[i] = torch.Tensor([[0., 0., 1., 1.]]).float().cuda()
                labels[i] = torch.Tensor([0]).cuda()
            #print(labels[i])
            if len(labels[i]) == 1:
                print(labels[i][0])
                #labels[i] = torch.tensor([0])
                
            nbr_boxes = boxes[i].size(0)
            
            ious = IOUs(boxes[i], self.uncentered_default)
            db_max_ious_value, db_max_ious_box = ious.max(dim=0)
            box_max_ious_value, box_max_ious_db = ious.max(dim=1)
            db_max_ious_box[box_max_ious_db] = torch.LongTensor(range(nbr_boxes)).cuda()
            db_max_ious_value[box_max_ious_db] = self.threshold
            db_max_ious_label = labels[i][db_max_ious_box] 
            db_max_ious_label[db_max_ious_value < self.threshold] = 0 
            
            #print('DB' , (ground_truth_locs[i].float() == np.nan).sum())
            ground_truth_classes[i] = db_max_ious_label
            ground_truth_boxes[i] = deviate(center(boxes[i][db_max_ious_box]), self.default_boxes) 
            
        ground_truth_boxes.cuda()    
        ground_truth_classes.cuda()  
        L1_Loss = nn.L1Loss()
        positive_db = ground_truth_classes != 0
        box_loss = L1_Loss(predicted_boxes[positive_db], ground_truth_boxes[positive_db])
        #print('LOSS' , box_loss)
        nbr_classes = predicted_labels.size(2)
        nbr_positives = positive_db.sum(dim=1)
        #print(predicted_labels.view(-1, nbr_classes).size())
        
        
        #print(ground_truth_classes.view(-1).size())
        closs = self.CELoss(predicted_labels.view(-1, nbr_classes).float(), ground_truth_classes.view(-1).long())  
        closs = closs.view(batch_size, nbr_default)
        #print(positive_db.size())
        #print('done')
        positive_closs = closs[positive_db]
        neg_closs = closs.clone()  
        neg_closs[positive_db] = 0.
        neg_closs, _ = neg_closs.sort(dim=1, descending=True)

        hardness_ranks = torch.Tensor(range(nbr_default)).unsqueeze(0).expand_as(neg_closs).cuda()
        hard_negatives = hardness_ranks < self.ratio * nbr_positives.unsqueeze(1)
        hardneg_closs = neg_closs[hard_negatives]

        closs = (hardneg_closs.sum() + positive_closs.sum()) / nbr_positives.sum()
       # print('conf loss' , conf_loss)
        #print('loc loss' , loc_loss)
        
        loss = closs + self.alpha * box_loss

        return loss

    