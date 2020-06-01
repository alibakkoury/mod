import time
import torch.utils.data
from networks import ObjectDetection_SSD, LossFunction
import argparse

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default = "SSD300")
    parser.add_argument("--batch_size", default = 32)
    parser.add_argument("--nb_epochs" , default = 200000)
    parser.add_argument("--lr" , default = 0.0001)
    parser.add_argument("--display_count" , default = 100)
    parser.add_argument("--nbr_classes" , default = 500)
    parser.add_argument("--checkpoint" , default = 50000)
    parser.add_argument("--checkpoint_dir" , default = "checkpoints")
    opt = parser.parse_args()
    return opt

def train(model , opt , train_loader):
    
    model.cuda()
    model.train()

    # criterion
    box = model.create_boxes()
    criterion = LossFunction(box)
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    
    for epoch in range(opt.nb_epochs):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()
        
        img = inputs['img']
        boxes = inputs['GT']
        label = inputs['label']

        predicted_boxes , predicted_scores = model(img)

        loss = criterion( predicted_boxes , predicted_scores , boxes , label )    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % opt.display_count == 0:
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f' % (epoch+1, t, loss.item()), flush=True)

        if (epoch+1) % opt.checkpoint == 0:
            torch.save(model.cpu().state_dict(), opt.checkpoint_dir)
            model.cuda()



def main():
    opt = get_opt()
    train_dataset = CPDataset(opt)
    train_loader = CPDataLoader(opt, train_dataset)

    model = ObjectDetection_SSD(nbr_classes = opt.nbr_classes)

    train(model , opt , train_loader)
        
    print('Finished training')