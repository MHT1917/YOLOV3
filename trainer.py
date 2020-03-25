from dataset import MyDataset
from darknet53 import *
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
import torch.optim as optim

class Trainer:
    def __init__(self):
        self.save_path = 'models/net.pt'
        self.net = MainNet().cuda()
        if os.path.exists(self.save_path):
            self.net.load_state_dict(torch.load(self.save_path))
        self.traindata = MyDataset()
        self.trainloader = DataLoader(self.traindata, batch_size=4, shuffle=True)
        self.bceloss = nn.BCELoss()
        self.mseloss = nn.MSELoss()
        self.nllLoss = nn.NLLLoss()
        self.optimizer = optim.Adam(self.net.parameters())

    def loss_fn(self,output, target, alpha):
        output = output.permute(0, 2, 3, 1)
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)

        output1 = torch.sigmoid(output[...,:3])
        output2 = torch.log_softmax(output[...,5:],dim=1)

        mask_obj = target[..., 0] > 0
        mask_noobj = target[..., 0] == 0
        loss_obj_cls_offset = self.bceloss(output1[mask_obj],target[mask_obj][:,:3].float())
        loss_obj_w_h = self.mseloss(output[mask_obj][:,3:5],target[mask_obj][:,3:5].float())
        loss_obj_category = self.nllLoss(output2[mask_obj],target[mask_obj][:,5].long())

        loss_obj = (loss_obj_cls_offset+loss_obj_w_h+loss_obj_category)/3
        loss_noobj = self.bceloss(output1[mask_noobj][:,0],target[mask_noobj][:,0].float())
        loss = alpha * loss_obj + (1 - alpha) * loss_noobj
        return loss
    def trainer(self):
        # self.net.train()
        epoch = 0
        while True:
            print(epoch)
            for target_13, target_26, target_52, img_data in self.trainloader:
                target_13, target_26, target_52, img_data = target_13.cuda(), target_26.cuda(), target_52.cuda(), img_data.cuda()
                output_13, output_26, output_52 = self.net(img_data)
                loss_13 = self.loss_fn(output_13, target_13, 0.9)
                loss_26 = self.loss_fn(output_26, target_26, 0.9)
                loss_52 = self.loss_fn(output_52, target_52, 0.9)

                loss = loss_13 + loss_26 + loss_52
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print(loss.item())
                # torch.save(self.net.state_dict(), self.save_path)
            if epoch%10==0:
                torch.save(self.net.state_dict(), 'models/{}net.pt'.format(epoch))
            epoch += 1
if __name__ == '__main__':
    obj = Trainer()
    obj.trainer()