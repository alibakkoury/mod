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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg16_bn' : 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
}


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


def xy_to_cxcy(xy):
    """
    Convert bounding boxes from boundary coordinates (x_min, y_min, x_max, y_max) to center-size coordinates (c_x, c_y, w, h).
    :param xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h


def cxcy_to_xy(cxcy):
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h


class ObjectDetection_SSD(nn.Module):
  def __init__(self, nbr_classes = 1000): #Initialisation du système SSD
    
    super(ObjectDetection_SSD, self).__init__() 

    self.cnn = vgg16(nbr_classes )  #Base du réseau VGG16, sans la partie Dense, renvoie la sortie de 2 layers
    self.box = BoxRegressionNet()  #Réseau de génération des rectangles (Regression)
    self.pred = ClassificationNet(nbr_classes) #Réseau de classification, renvoie les localisations des rectangles et les prédictions pour les nbr_classes classes pour chacun d'eux
    self.prior_boxes = self.create_boxes().cuda()  
    self.nbr_classes = nbr_classes

  def forward(self , x):

    feature_map1 , feature_map2 = self.cnn(x)
    feature_map3 , feature_map4, feature_map5, feature_map6 = self.box(feature_map2) 
    boxes , scores = self.pred(feature_map1 , feature_map2 ,feature_map3 ,feature_map4 ,feature_map5 ,feature_map6)

    return boxes , scores

  def create_boxes(self):
    
      fmap_dims = {'conv4_3': 38,
                     'conv7': 19,
                     'conv8_2': 10,
                     'conv9_2': 5,
                     'conv10_2': 3,
                     'conv11_2': 1}

      obj_scales = {'conv4_3': 0.1,
                      'conv7': 0.2,
                      'conv8_2': 0.375,
                      'conv9_2': 0.55,
                      'conv10_2': 0.725,
                      'conv11_2': 0.9}

      aspect_ratios = {'conv4_3': [1., 2., 0.5],
                         'conv7': [1., 2., 3., 0.5, .333],
                         'conv8_2': [1., 2., 3., 0.5, .333],
                         'conv9_2': [1., 2., 3., 0.5, .333],
                         'conv10_2': [1., 2., 0.5],
                         'conv11_2': [1., 2., 0.5]} 

      fmaps = list(fmap_dims.keys())

      prior_boxes = []

      for k, fmap in enumerate(fmaps):
          for i in range(fmap_dims[fmap]):
              for j in range(fmap_dims[fmap]):
                  cx = (j + 0.5) / fmap_dims[fmap]
                  cy = (i + 0.5) / fmap_dims[fmap]

                  for ratio in aspect_ratios[fmap]:
                      prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])
                      if ratio == 1.:
                          try:
                              additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            # For the last feature map, there is no "next" feature map
                          except IndexError:
                              additional_scale = 1.
                          prior_boxes.append([cx, cy, additional_scale, additional_scale])

      prior_boxes = torch.FloatTensor(prior_boxes) 
      prior_boxes.clamp_(0, 1) 

      return prior_boxes
  
  def NMS_1(self , class_scores , decoded_locs , n_boxes, min_score): #On se débarrasse des prédictions avec un score n'atteignant pas un seuil choisi
    res = class_scores
    index = []
    i = 0
    j = 0
    while(j < n_boxes):    
      if res[i] < min_score :
         res = torch.cat([res[:i] , res[i+1:]])
         
      else : 
        i+=1
        index.append(j)
      j+=1
    index = torch.LongTensor(index)
    decoded_locs = decoded_locs[index] 

    res, sort_ind = res.sort(dim=0, descending=True)
    decoded_locs = decoded_locs[sort_ind]
    

    return res , decoded_locs

  def find_intersection(self , set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.
    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


  def find_jaccard_overlap(self , set_1, set_2):

    # Find intersections
    intersection = self.find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)


  def IoU(self , t1 , t2):

      def IoU_rect(rect1,rect2): #On calcule le IoU entre 2 rectangles : 
          x1,y1,w1,h1 = tuple(rect1.tolist())
          x2,y2,w2,h2 = tuple(rect2.tolist())
          xA,yA = max(x1,x2),max(y1,y2)
          xB,yB = min(x1+w1,x2+w2),min(y1+h1,y2+h2)
          inter = max(0,xB - xA)*max(0,yB-yA)
          union = w1*h1 + w2*h2 - inter
          return inter/union

      n1 = t1.size()[0]
      n2 = t2.size()[0]  
      res = torch.zeros((n1 , n2))

      for i in tqdm(range(n1)):
        for j in tqdm(range(n2)):
          res[i,j] = IoU_rect(t1[i] , t2[j])

      return res

  def NMS_2(self , n_boxes , iou , top_k ):

    boxes = []
    scores = []
    labels = []
    
    is_Valid = torch.zeros(n_boxes).cuda()

    for i in range(n_boxes):
        if is_Valid[i] : 
          continue
          
        is_Valid = torch.max(is_Valid, iou[box] > max_overlap)

        is_Valid[i] = 0
        boxes.append(class_decoded_locs[1 - isValid])
        labels.append(torch.LongTensor((1 - isValid).sum().item() * [c]))
        scores.append(class_scores[1 - isValid])
        
    if len(boxes) == 0:
       boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]))
       labels.append(torch.LongTensor([0]))
       scores.append(torch.FloatTensor([0.]))

    boxes = torch.cat(boxes, dim=0)  
    labels = torch.cat(labels, dim=0) 
    scores = torch.cat(scores, dim=0)  
    n_objects = scores.size()[0]

    if n_objects > top_k:
      scores, sort_ind = scores.sort(dim=0, descending=True)
      scores = scores[:top_k]  # (top_k)
      boxes = boxes[sort_ind][:top_k]  # (top_k, 4)
      labels = labels[sort_ind][:top_k]  # (top_k)

    return boxes , scores , labels
        

  def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):

    res_boxes = []
    res_scores = []
    res_labels = []

    n = predicted_scores.size()[0] #Taille du batch
    predicted_scores = F.softmax(predicted_scores, dim=2)

    for k in tqdm(range(n)):#On parcourt le batch
      decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[k], self.prior_boxes)) 
      
      boxes = []
      scores = []
      labels = []

      max_scores, best_label = predicted_scores[k].max(dim=1)  #Liste de la taille du nombre de priors

      for i in tqdm(range(self.nbr_classes)):
        
        class_scores = predicted_scores[k][:, i]

        n_boxes = class_scores.size()[0]

        class_scores , index = self.NMS_1(class_scores , decoded_locs , n_boxes , min_score)

        n_boxes = class_scores.size()[0]

        iou = self.find_jaccard_overlap(decoded_locs , decoded_locs)

        boxes , scores , labels = self.NMS_2(n_boxes , iou , top_k)
 
        res_boxes.append(boxes)
        res_scores.append(scores)
        res_labels.append(labels)

    return res_boxes , res_scores , res_labels

def find_jaccard_overlap(set_1, set_2):
    # Find intersections
    intersection = find_intersection(set_1, set_2)# (n1, n2)
    intersection.cpu()

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)
    
    areas_set_1.cpu()
    areas_set_2.cpu()

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = (areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection).cpu()  # (n1, n2)

    return intersection / union  # (n1, n2)
    
def find_intersection(set_1, set_2):
    set_1.cpu()
    set_2.cpu()
    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].cpu().unsqueeze(1), set_2[:, :2].cpu().unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].cpu().unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)

class LossFunction(nn.Module):
    """
    La MultiBoxloss est notre loss d'apprentissage, elle est constitué d'une loss de localisation et d'une loss de classification.
    """

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        
        super(LossFunction, self).__init__()
        
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.l1Loss = nn.L1Loss()
        self.CELoss = nn.CrossEntropyLoss(reduce=False)

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        
        #boxes = list(boxes)
        #labels = list(labels)
        
        batch_size = predicted_locs.size(0)
        nbr_priors = self.priors_cxcy.size(0)
        print('predicted_scores' , predicted_scores.size())
        print('nbr_priors' , nbr_priors)
        nbr_classes = predicted_scores.size(2)
        


        ground_truth_locs = torch.zeros((batch_size, nbr_priors, 4)).float().cuda()  
        ground_truth_classes = torch.zeros((batch_size, nbr_priors)).cuda()
        print(ground_truth_classes.size())

        for i in range(batch_size):
            nbr_boxes = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i], self.priors_xy)

            # on identifie pour chaque prior la boxe qui lui correspond le plus
            prior_max_overlap_value, prior_max_overlap_box = overlap.max(dim=0)

            # on veut que chaque boxe soit associée a au moins un prior positif (IoU>=seuil)
            # on va donc pour chaque boxe choisir le prior qui lui correspond le plus et les associer avec un IoU arbitraire fixé au seuil  
            box_max_overlap_value, box_max_overlap_prior = overlap.max(dim=1)
            prior_max_overlap_box[box_max_overlap_prior] = torch.LongTensor(range(nbr_boxes))
            prior_max_overlap_value[box_max_overlap_prior] = self.threshold

            # on associe a chaque prior le label de la boxe qui lui correspond
            # si le prior n'est pas suffisamment proche d'aucune boxe on le classifie comme ne contenant aucun objet
            prior_max_overlap_label = labels[i][prior_max_overlap_box] 
            prior_max_overlap_label[prior_max_overlap_value < self.threshold] = 0 

            ground_truth_classes[i] = prior_max_overlap_label
            ground_truth_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][prior_max_overlap_box]), self.priors_cxcy) 
        ground_truth_locs.cuda()    
        ground_truth_classes.cuda()
            
        

        positive_priors = ground_truth_classes != 0

        # LOCALIZATION LOSS

        loc_loss = self.l1Loss(predicted_locs[positive_priors], ground_truth_locs[positive_priors])

        # CONFIDENCE LOSS

        nbr_positives = positive_priors.sum(dim=1)
        nbr_hard_negatives = self.neg_pos_ratio * nbr_positives
        
        print(predicted_scores.view(-1, nbr_classes).size())
        print(ground_truth_classes.view(-1).size())
        

        conf_loss_all = self.CELoss(predicted_scores.view(-1, nbr_classes).float().cuda(), ground_truth_classes.view(-1).long()).cuda()  
        conf_loss_all = conf_loss_all.view(batch_size, nbr_priors).cpu()
        print(positive_priors.size())
        
        

        conf_loss_pos = conf_loss_all[positive_priors]

        conf_loss_neg = conf_loss_all.clone()  
        conf_loss_neg[positive_priors] = 0.
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)

        hardness_ranks = torch.Tensor(range(nbr_priors)).unsqueeze(0).expand_as(conf_loss_neg).cuda()
        hard_negatives = hardness_ranks < nbr_hard_negatives.unsqueeze(1)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]

        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / nbr_positives.sum()
        conf_loss.cuda()

        multiboxloss = conf_loss + self.alpha * loc_loss

        return multiboxloss

