# coding: utf-8
# In[1]:
from __future__ import division
import pandas as pd 
import numpy as np 
#import matplotlib.pyplot as plt 
#get_ipython().magic(u'matplotlib inline')
import torch 
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import argparse 

from mydataset import *
from model import * 

def main():
    parser = argparse.ArgumentParser() 

    parser.add_argument('--weight-decay', type=float, default=1e-3,
                        help='weight decay')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint',
                        help='logdir')
    parser.add_argument('--schedule', type=int, nargs='+', default=[20, 30, 45],
                        help='Decreasing learning rate at these epochs')
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=5e-3)
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--test-only', dest='test_only', action='store_true')
    parser.add_argument('--test-eval', type = bool, default=False)
    
    args = parser.parse_args()

    cwd = os.getcwd() 

    check_point = os.path.join(cwd, args.checkpoint)
    if not os.path.exists(check_point):
        os.makedirs(check_point)
    model_path = os.path.join(check_point, 'best_model.pt')
    writer = SummaryWriter(check_point)

    trainset = MyDataset('train')
    valset = MyDataset('valid')

    # In[6]:

    trainloader = DataLoader(dataset=trainset, shuffle=True, batch_size=args.batch_size)
    valloader = DataLoader(dataset=valset, shuffle=False, batch_size=args.batch_size)
    model = Net()
    # if args.test_only:
    #     model.load_state_dict(torch.load(model_path))
    model.cuda()
    optimizer = optim.SGD(model.parameters(), momentum=0.95, lr=args.learning_rate, weight_decay=args.weight_decay)
    cls_weight = np.array([600, 4800, 3000, 1600, 3600, 2000, 700, 1200, 600, 500])
    cls_weight = 4800./cls_weight
#    cls_weight = cls_weight/np.linalg.norm(cls_weight)
    cls_weight = torch.from_numpy(cls_weight)
    cls_weight = cls_weight.type(torch.FloatTensor).cuda()
    loss_fn = nn.CrossEntropyLoss(weight=None)
    scheduler = lr_scheduler.MultiStepLR(optimizer, args.schedule, gamma=0.2)

    error_cnt = np.zeros(10)
    if args.test_only:
        val_acc = 0.0
        running_loss = 0.
        for i, (inputs, labels) in enumerate(valloader):
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = model(inputs)
            labels = labels.squeeze()
            loss = loss_fn(outputs, labels)
            running_loss += loss.data[0]
            _, preds = outputs.data.topk(1, 1, True, True)
            preds = preds.t()
            corrects = preds.eq(labels.data.view(1, -1).expand_as(preds)).squeeze()
            val_acc += torch.sum(corrects)
            labels = labels.data.cpu().numpy()
            corrects = corrects.cpu().numpy()
            error_cnt += np.bincount(labels[corrects==0],minlength=10)
        print(error_cnt)

        val_acc = val_acc.cpu().numpy()/len(valset)
        val_loss = args.batch_size*running_loss/len(valset)
        print('Validation accuracy:' + str(val_acc))
        print('Validattion loss: ' + str(val_loss))
        return

    # In[7]:
    best_val_acc = 0
    for epoch in range(args.num_epochs):
        running_loss = 0.0
        train_acc = 0 
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = model(inputs)
            labels = labels.squeeze()
            optimizer.zero_grad()
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]

            _, preds = outputs.data.topk(1, 1, True, True)
            preds = preds.t()
            corrects = preds.eq(labels.data.view(1, -1).expand_as(preds))
            train_acc += torch.sum(corrects)

        lr = optimizer.param_groups[0]['lr'] 
        writer.add_scalar('Learning rate', lr, epoch)

        train_acc = train_acc.cpu().numpy()/len(trainset) 
        train_loss = args.batch_size*running_loss/len(trainset)
        writer.add_scalar('Train loss', train_loss, epoch)
        writer.add_scalar('Train acc', train_acc, epoch)

        print('Training loss: ' + str(train_loss))
        print('Learning rate: ' + str(lr))

        scheduler.step()
        val_acc = 0.0
        running_loss = 0.
        for i, (inputs, labels) in enumerate(valloader):
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = model(inputs)
            labels = labels.squeeze()
            loss = loss_fn(outputs, labels)
            running_loss += loss.data[0]
            _, preds = outputs.data.topk(1, 1, True, True)
            preds = preds.t()
            corrects = preds.eq(labels.data.view(1, -1).expand_as(preds)).squeeze()
            val_acc += torch.sum(corrects) 
        val_acc = val_acc.cpu().numpy()/len(valset)
        val_loss = args.batch_size*running_loss/len(valset)
        writer.add_scalar('Val acc', val_acc, epoch)
        writer.add_scalar('Val loss', val_loss, epoch)
        print('Validation accuracy:' + str(val_acc))
        print('Validattion loss: ' + str(val_loss))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
    json_file = os.path.join(check_point, 'logfile.json') 
    writer.export_scalars_to_json(json_file)
    writer.close()

    if args.test_eval:
        model.load_state_dict(torch.load(model_path))
        testset = MyDataset('test')
        testloader = DataLoader(dataset=testset, shuffle=False, batch_size=1)
        test  = np.zeros(len(testset))
        model.eval()
        for i, inputs in enumerate(testloader):
            inputs  = Variable(inputs.cuda())
            outputs = model(inputs)
            outputs = outputs.detach().cpu().numpy()
            test[i] = np.argmax(outputs)

        with open('test.csv','w') as f:
            f.write('id,class\n')
            for i,pre in enumerate(test):
                f.write(str(i+1)+','+str(int(pre)+1) + '\n')
        print('Submission file has been written!')
    
if __name__=='__main__':
    main()
    
        
