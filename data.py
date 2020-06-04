import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision

from PIL import Image
from PIL import ImageDraw

import numpy as np
import csv





class Dataset():
    def __init__(self , coco):
        super(Dataset, self).__init__()
        self.resize = torchvision.transforms.Resize((300,300))
        self.topil = torchvision.transforms.ToPILImage()
        self.transform = torchvision.transforms.ToTensor()
        self.compose = torchvision.transforms.Compose([self.topil , self.resize , self.transform])
        self.coco = coco
        
    def get_boxes_labels(self , i):
        transform = torchvision.transforms.ToTensor()
        
        
        n = self.coco.__len__()
        boxes = []
        labels = []
        data = self.coco.__getitem__(i)
        m = len(data[1])
        img = data[0][:]
        p , w , h = img.size()
        t = torch.tensor([1/h , 1/w , 1/h , 1/w])
        for k in range(m):
            box = torch.tensor(data[1][k]['bbox'])*t
            label = data[1][k]['category_id']
            box = box.numpy()
            boxes.append(box)
            labels.append(label)
            
            
            
        boxes = torch.tensor(boxes).cuda()
        labels = torch.tensor(labels).cuda()
        return boxes , labels
    
    def __len__(self):
        return self.coco.__len__()
    
    
    def __getitem__(self, index):
        
        data = self.coco.__getitem__(index) 
        img = data[0][:]
        img = self.compose(img)
        
        boxes , labels = self.get_boxes_labels(index)

        result = {
            'img' : img,
            'boxes' : boxes,
            'labels' : labels,
        }

        return img , boxes , labels
    
        
    def collate_fn(self, batch):
    
        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)

        return images, boxes, labels
    

    

class DataLoader(object):
    def __init__(self, opt, dataset):
        super(DataLoader, self).__init__()

        self.data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=opt.batch_size ,collate_fn = dataset.collate_fn)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()
       
    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch
    

        




    

