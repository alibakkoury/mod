import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision

from PIL import Image
from PIL import ImageDraw

import numpy as np
import csv

def dataset(opt):
     data = torchvision.datasets.CocoDetection(opt.traindata_dir + 'images/' , opt.traindata_dir + 'ann/' )
     return data

class DataLoader(object):
    def __init__(self, opt, dataset):
        super(DataLoader, self).__init__()

        self.data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=opt.batch_size)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()
       
    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch
        




    

