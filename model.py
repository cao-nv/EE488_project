import torch 
import torch.nn as nn


class Net(nn.Module): 
    def __init__(self):
        super(Net, self).__init__() 
        modules = []
        for i in range(4):
            modules.append(nn.Linear(128, 128))
            modules.append(nn.ReLU(True))
            
        modules.append(nn.Linear(128, 64))
        modules.append(nn.ReLU(True))
        modules.append(nn.Linear(64, 10))
        self.body = nn.Sequential(*modules)
    
    def forward(self, x):
        return self.body(x)

