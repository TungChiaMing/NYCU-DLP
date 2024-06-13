import torch.nn as nn
import model.utils as utils

class DeepConvNet(nn.Module):
    def __init__(self, activation='elu'):
        super().__init__()

        self.C = 2 # number of channels
        self.T = 750 # number of time points
        self.N = 2 # number of classes

        self.activation = utils.activate(activation)

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1,5), padding=0),
            nn.Conv2d(25, 25, kernel_size=(self.C,1)),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1),
            self.activation,
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1,5), padding=0),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.1),
            self.activation,
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1,5), padding=0),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1),
            self.activation,
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5),
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1,5), padding=0),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1),
            self.activation,
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5),
        )

        self.flatten = nn.Flatten() 

        self.fc = nn.Sequential(
            nn.Linear(in_features=8600, out_features=self.N, bias=True),
        )

      

    def forward(self, x):
        pred = self.conv_block1(x)
        pred = self.conv_block2(pred)
        pred = self.conv_block3(pred)
        pred = self.conv_block4(pred)

        pred = self.flatten(pred)
        # print(pred.size())
        pred = self.fc(pred)
        return pred
    

    def compile(self, loss, optimizer, lr=1e-2):
        self.optimizer = optimizer
        self.loss = loss
        self.lr = lr

#model = DeepConvNet()
#print(model)