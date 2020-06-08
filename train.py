import time
import torch.utils.data
from networks import ObjectDetection_SSD, LossFunction, default_boxes, undeviate, IOUs
from data import DataLoader , Dataset
import argparse
import os 
import torchvision
import cv2
import numpy as np
from PIL import Image

from tensorboardX import SummaryWriter
from visualization import board_add_image, board_add_images

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default = "SSD300")
    parser.add_argument("--batch_size", default = 20)
    parser.add_argument("--nb_epochs" , default = 50000)
    parser.add_argument("--lr" , default = 0.0001)
    parser.add_argument("--display_count" , default = 100)
    parser.add_argument("--nbr_classes" , default = 91)
    parser.add_argument("--checkpoint" , default = 1000)
    parser.add_argument("--checkpoint_dir" , default = 'checkpoints/')
    parser.add_argument("--traindata_dir" , default = 'data/train/')
    
    opt = parser.parse_args()
    return opt

topil = torchvision.transforms.ToPILImage()
totensor = torchvision.transforms.ToTensor()



    
def IoU(rect1,rect2):
  x1,y1,w1,h1 = rect1
  x2,y2,w2,h2 = rect2
  xA,yA = max(x1,x2),max(y1,y2)
  xB,yB = min(x1+w1,x2+w2),min(y1+h1,y2+h2)
  inter = max(0,xB - xA)*max(0,yB-yA)
  union = w1*h1 + w2*h2 - inter
  return inter/union

def get_labeled_boxes(predicted_boxes , predicted_scores , box , threshold=.5,iou=0.5, ioug = 0.3):
    boxes = [] 
    scores = []
    labels = []
    rects = predicted_boxes
    preds = predicted_scores
    #print('SIZE' , rects.size() , box.size())
    ious = IOUs(box, rects)
    db_max_ious_value, db_max_ious_box = ious.max(dim=0)
    rects = rects[db_max_ious_box[db_max_ious_value > ioug]]
    n = len(rects)
    for i in range(n) : 
        cx , cy , w , h = rects[i]*300
        x , y = cx-w/2 , cy-h/2
        if (preds[i].max().item() > threshold):
            cond = True
            k = len(boxes)
            for l in range(k) :
                if (IoU(boxes[l] , (x,y,w,h)) > iou) and labels[l]==predicted_scores[i].argmax():
                    cond = False
                    break
            if cond : 
                boxes.append((x,y,w,h))
                labels.append(predicted_scores[i].argmax().item())
                scores.append(predicted_scores[i].max().item())
    return boxes, scores, labels


def showrects(img,rects,maxRect=3000):
    imag = img.cpu()
    imag = topil(imag)
    #print(imag.size)
    imOut = cv2.cvtColor(np.float32(imag),cv2.COLOR_RGB2BGR)
    #imOut = np.float32(imag)
    #print(rects.size())
    for i,rect in enumerate(rects):
        if (i < maxRect):
            x, y, w, h = rect*300
            cv2.rectangle(imOut, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 2, cv2.LINE_AA)
    
    #image = imOut
    image = cv2.cvtColor(np.float32(imOut),cv2.COLOR_BGR2RGB)
    #image = Image.fromarray(image)
    image = totensor(imOut).view((1,3,300,300))
    #print("Done")
    return imOut , image


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
    


def train(model , opt , train_loader , board):
    
    print("Training ...")
    
    model.cuda()
    model.train()

    # criterion
    box = default_boxes()
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
        #print('LOSS' , loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        #print('epoch done')

        if (epoch+1) % opt.display_count == 0:
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f' % (epoch+1, t, loss.item()), flush=True)

        if (epoch+1) % opt.checkpoint == 0:
            torch.save(model.cpu().state_dict(), opt.checkpoint_dir + 'check.pth')
            model.cuda()
       
        if (epoch+1) % opt.display_count == 0:
            board.add_scalar('metric', loss.item(), epoch+1)
            #image = img[0] 
            #box = boxes[0]
            #score = scores[0]
            #label = labels[0]
                
            #image_1 = show_objs(img_1 , boxes_1 , scores_1 , labels_1).squeeze()
                
            #print(image_1.size())
            #predicted_boxes[0] = undeviate(predicted_boxes[0] , box)

            n_batch = predicted_scores.size(0)

            n_b , m_b = predicted_boxes[0].size()
            
            #n_s , m_s = predicted_scores[0].size()
         
            boxes , labels , scores = model.detect_objects(predicted_boxes , predicted_scores , 0.1 , 0.45 , 5)

            images = []
            imOuts = []
            for i in range(n_batch) :
                #print(box[0].size())
                box  = boxes[i]
                imOut , image = showrects(img[i] , box) 
                #image = image.unsqueeze(0)
                images.append(image)
                imOuts.append(imOut)
            print(images[0].size())

            images = torch.cat(images , dim = 0)
            print(images.size())
        
            board_add_image(board, 'combine', images, epoch+1)
            string = 'images/step' + str(epoch) + 'jpg'
            cv2.imwrite("train_res/test{0}.jpg".format(epoch+1) , imOuts[0])
            
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f' % (epoch+1, t, loss.item()), flush=True)
                
            #image_1 = image_1.numpy()
            #cv2.imwrite('test/step_%.jpg'%(epoch) , image_1)
                
                
def main():
    opt = get_opt()
    
    transform = torchvision.transforms.ToTensor()

    coco = torchvision.datasets.CocoDetection('data/train2014/' , 'data/annotations/instances_train2014.json' , transform=transform)
    
    if not os.path.exists('tensorboard'):
        os.makedirs('tensorboard')
    board = SummaryWriter(log_dir = os.path.join('tensorboard', 'SSD'))

    if not os.path.exists('checkpoints'):
               os.makedirs('checkpoints')
    
    if not os.path.exists('train_res'):
               os.makedirs('train_res')
    
    train_dataset = Dataset(coco)
    train_loader = DataLoader(opt, train_dataset)

    model = ObjectDetection_SSD(nbr_classes = opt.nbr_classes)

    #if os.path.exists("check.pth"):
     #  model.load_state_dict(torch.load("check.pth"))
    
    start = time.time()

    train(model , opt , train_loader , board)
    
    end = time.time()
    
    duree = end-start
    
    torch.save(model.cpu().state_dict(), 'model.pth')
    
    print('Finished training in' , duree)

main()