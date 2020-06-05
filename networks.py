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
    model = VGG(make_layers(cfg['SSD1'], batch_norm=False),make_layers(cfg['SSD2'],  depth = 512,batch_norm=False, SSD = True), **kwargs)
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

    self.cnn = vgg16(nbr_classes )  #Base du réseau VGG16, sans la partie Dense, renvoie la sortie de 2 layers
    self.box = BoxRegressionNet()  #Réseau de génération des rectangles (Regression)
    self.pred = ClassificationNet(nbr_classes) #Réseau de classification, renvoie les localisations des rectangles et les prédictions pour les nbr_classes classes pour chacun d'eux

  def forward(self , x):

    feature_map1 , feature_map2 = self.cnn(x)
    feature_map3 , feature_map4, feature_map5, feature_map6 = self.box(feature_map2) 
    boxes , scores = self.pred(feature_map1 , feature_map2 ,feature_map3 ,feature_map4 ,feature_map5 ,feature_map6)

    return boxes , scores




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

    