import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.misc
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import time
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import cv2

device = torch.device('cuda')

print('Analysis 1 Comparison')

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

class dataset(Dataset):
    def __init__(self, df_path, train = False):
        self.df = pd.read_csv(df_path)
        self.train = train
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        imgage_folder = '/scratch/cz2064/myjupyter/BDML/Project/Data/data_1024/'
        file_id = self.df.iloc[idx]['File ID']
        file_name = self.df.iloc[idx]['File Name']
        image_path = imgage_folder + file_id + '/' + file_name[:-4] + '_1024.jpg'
        image = Image.open(image_path)
        image_tensor = preprocess(image)
        label = self.df.iloc[idx]['label']
        sample = {'x': image_tensor, 'y': label, 'id':file_id}
    
        return sample       

train_df_path = '/scratch/cz2064/myjupyter/BDML/Project/Phase5/Train_Test_Split/train.csv'
val_df_path = '/scratch/cz2064/myjupyter/BDML/Project/Phase5/Train_Test_Split/val.csv'
test_df_path = '/scratch/cz2064/myjupyter/BDML/Project/Phase5/Train_Test_Split/test.csv'


BATCH_SIZE = 8
train_loader = DataLoader(dataset(train_df_path), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset(val_df_path), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset(test_df_path), batch_size=BATCH_SIZE, shuffle=True)


VGG_11 = torch.hub.load('pytorch/vision:v0.6.0', 'vgg11_bn', pretrained=True)
class VGG_CAM(nn.Module):
    def __init__(self, features = VGG_11.features, n_classes = 3):
        super(VGG_CAM, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, n_classes,bias=False)
            
    def forward(self, x):
        x = self.features(x)
        self.featuremap1 = x.detach()
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
def train(model, train_loader=train_loader, val_loader=val_loader, learning_rate=0.0001, num_epoch=100):
    start_time = time.time()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    train_loss_return = []
    train_acc_return = []
    val_loss_return = []
    val_acc_return = []
    best_acc = 0
    
    for epoch in range(num_epoch):
        # Training steps
        correct = 0
        total = 0
        predictions = []
        truths = []
        model.train()
        train_loss_list = []
        for i, (sample) in enumerate(train_loader):
            data = sample['x'].to(device)
            labels = sample['y'].to(device)
            outputs = model(data)
            pred = outputs.data.max(-1)[1]
            predictions += list(pred.cpu().numpy())
            truths += list(labels.cpu().numpy())
            total += labels.size(0)
            correct += (pred == labels).sum()
            model.zero_grad()
            loss = loss_fn(outputs, labels)
            train_loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
        # report performance
        acc = (100 * correct / total)
        train_acc_return.append(acc)
        train_loss_every_epoch = np.average(train_loss_list)
        train_loss_return.append(train_loss_every_epoch)
        print('----------Epoch{:2d}/{:2d}----------'.format(epoch+1,num_epoch))
        print('Train set | Loss: {:6.4f} | Accuracy: {:4.2f}% '.format(train_loss_every_epoch, acc))
        
        # Evaluate after every epochh
        correct = 0
        total = 0
        model.eval()
        predictions = []
        truths = []
        val_loss_list = []
        with torch.no_grad():
            for i, (sample) in enumerate(val_loader):
                data = sample['x'].to(device)
                labels = sample['y'].to(device)
                outputs = model(data)
                loss = loss_fn(outputs, labels)
                val_loss_list.append(loss.item())
                pred = outputs.data.max(-1)[1]
                predictions += list(pred.cpu().numpy())
                truths += list(labels.cpu().numpy())
                total += labels.size(0)
                correct += (pred == labels).sum()
            # report performance
            acc = (100 * correct / total)
            val_acc_return.append(acc)
            val_loss_every_epoch = np.average(val_loss_list)
            val_loss_return.append(val_loss_every_epoch)
            if acc > best_acc:
                best_acc = acc
                best_model_wts = model.state_dict()
            save_model(model,train_loss_return,train_acc_return,val_loss_return,val_acc_return,best_model_wts)
            elapse = time.strftime('%H:%M:%S', time.gmtime(int((time.time() - start_time))))
            print('Test set | Loss: {:6.4f} | Accuracy: {:4.2f}% | time elapse: {:>9}'\
                  .format(val_loss_every_epoch, acc,elapse))
    return model,train_loss_return,train_acc_return,val_loss_return,val_acc_return,best_model_wts

def save_model(model,train_loss_return,train_acc_return,val_loss_return,val_acc_return,best_model_wts):
    state = {'best_model_wts':best_model_wts, 'model':model, \
             'train_loss':train_loss_return, 'train_acc':train_acc_return,\
             'val_loss':val_loss_return, 'val_acc':val_acc_return}
    torch.save(state, 'checkpoint_Model.pt')
    return None    


model = VGG_CAM().to(device)
train(model,num_epoch=100)
