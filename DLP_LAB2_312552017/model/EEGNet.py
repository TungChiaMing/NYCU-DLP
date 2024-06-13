import torch.nn as nn
import model.utils as utils
import torch

class EEGNet(nn.Module):
    def __init__(self, activation='elu'):
        super().__init__()
        self.activation = utils.activate(activation)
 

        self.first_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),  
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            self.activation,
            nn.AvgPool2d(kernel_size=(1,4), stride=(1,4), padding=0),
            nn.Dropout(p=0.25),
        )    
        
        self.separable_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            self.activation,
            nn.AvgPool2d(kernel_size=(1,8), stride=(1,8), padding=0),
            nn.Dropout(p=0.25),
        )

        self.flatten = nn.Flatten() # flatten to (batch_size, num_channels*height*width)
        
        self.classify = nn.Sequential(
            nn.Linear(in_features=736, out_features=2, bias=True),    
            # input size 750 -> 736 caused by dep and sep layer
            # the number of output channels is 32, 
            # and the dimension of the output feature maps is 1x23.

        )


    
    def forward(self, x): 
        pred = self.first_conv(x)
        pred = self.depthwise_conv(pred)
        pred = self.separable_conv(pred)
        # print(pred.size())
        # (batch_size, num_channels, height, width)
        # torch.Size([64, 32, 1, 23])
        # torch.Size([56, 32, 1, 23])
        # torch.Size([64, 32, 1, 23])
        # torch.Size([64, 32, 1, 23])
        pred = self.flatten(pred)
        # print(pred.size())
        pred = self.classify(pred)
        return pred
         
    def compile(self, loss, optimizer, lr=1e-2):
        self.optimizer = optimizer
        self.loss = loss
        self.lr = lr


            
#model = EEGNet()
#print(model)


#model = EEGNet()

#total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#print(f'Total number of parameters in the model: {total_params}')
