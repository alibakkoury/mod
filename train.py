import time
import torch.utils.data
from networks import ObjectDetection_SSD, LossFunction
from data import DataLoader , Dataset
import argparse
import os 
import torchvision
import cv2

from tensorboardX import SummaryWriter
from visualization import board_add_image, board_add_images

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default = "SSD300")
    parser.add_argument("--batch_size", default = 20)
    parser.add_argument("--nb_epochs" , default = 10000)
    parser.add_argument("--lr" , default = 0.0001)
    parser.add_argument("--display_count" , default = 10)
    parser.add_argument("--nbr_classes" , default = 91)
    parser.add_argument("--checkpoint" , default = 50000)
    parser.add_argument("--checkpoint_dir" , default = 'checkpoints')
    parser.add_argument("--traindata_dir" , default = 'data/train/')
    
    opt = parser.parse_args()
    return opt


def show_objs(img ,boxes, scores , labels):
    
    n = len(boxes)
    im = img.copy()
    im = im.numpy()
    imOut = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    
  
    for i in range(n):
        rect = 300*boxes[i].numpy()
        x_min, y_min, x_max, y_max = rect
        cv2.rectangle(imOut, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imOut, "{} : {:.2f}".format(labels[label],score), (x_min,y_min+10),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,0,0))
    
    totensor = torchvision.transforms.ToTensor()
    imOut = totensor(imOut)
    
    return imOut
    


def train(model , opt , train_loader):
    
    print("Training ...")
    
    model.cuda()
    model.train()

    # criterion
    box = model.create_boxes()
    criterion = LossFunction(box).cuda()
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    
    #os.mkdir(opt.checkpoint_dir)
    
    for epoch in range(opt.nb_epochs):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()
        
        img = inputs[0].cuda()
        boxes = inputs[1]
        labels = inputs[2]
        

        #for i in range(opt.batch_size):
          #  boxes[i] = torch.cat(boxes[i] , dim=1)
           # labels[i] = torch.cat(labels[i], dim=1)

        predicted_boxes , predicted_scores = model(img)
        
        loss = criterion(predicted_boxes , predicted_scores , boxes , labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        print('epoch done')

        if (epoch+1) % opt.display_count == 0:
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f' % (epoch+1, t, loss.item()), flush=True)

        if (epoch+1) % opt.checkpoint == 0:
            torch.save(model.cpu().state_dict(), opt.checkpoint_dir)
            model.cuda()
       
        if (epoch+1) % opt.display_count == 0:
            boxes , scores , labels = model.detect_objects(predicted_boxes , predicted_scores , 0.01 , 0.45 , 200)
            if len(boxes)>4:
                img_1 = img[0]
                img_2 = img[1]
                img_3 = img[2]
                img_4 = img[3]
                
                
                boxes_1 = boxes[0]
                boxes_2 = boxes[1]
                boxes_3 = boxes[2]
                boxes_4 = boxes[3]
                
                scores_1 = scores[0]
                scores_2 = scores[1]
                scores_3 = scores[2]
                scores_4 = scores[3]
                
                labels_1 = labels[0]
                labels_2 = labels[1]
                labels_3 = labels[2]
                labels_4 = labels[3]
                
                image_1 = show_objs(img_1 , boxes_1 , scores_1 , labels_1).squeeze()
                image_2 = show_objs(img_2 , boxes_2 , scores_2 , labels_2).squeeze()
                image_3 = show_objs(img_3 , boxes_3 , scores_3 , labels_3).squeeze()
                image_4 = show_objs(img_4 , boxes_4 , scores_4 , labels_4).squeeze()
                
                
                visuals = torch.cat([image_1 , image_2 , image_3 , image_4])
                
                board_add_images(board, 'combine', visuals, step+1)
                board.add_scalar('metric', loss.item(), step+1)
                t = time.time() - iter_start_time
                print('step: %8d, time: %.3f, loss: %4f' % (step+1, t, loss.item()), flush=True)
                
                image_1 = image_1.numpy()
                cv2.imwrite('test/step_%.jpg'%(epoch) , image_1)
                
                
def main():
    opt = get_opt()
    
    transform = torchvision.transforms.ToTensor()

    coco = torchvision.datasets.CocoDetection('data/train2014/' , 'data/annotations/instances_train2014.json' , transform=transform)
    
    if not os.path.exists('tensorboard'):
        os.makedirs('tensorboard')
    board = SummaryWriter(log_dir = os.path.join('tensorboard', 'SSD'))
    
    train_dataset = Dataset(coco)
    train_loader = DataLoader(opt, train_dataset)

    model = ObjectDetection_SSD(nbr_classes = opt.nbr_classes)
    
    start = time.time()

    train(model , opt , train_loader)
    
    end = time.time()
    
    duree = end-start
    print('Finished training in' , duree)

main()