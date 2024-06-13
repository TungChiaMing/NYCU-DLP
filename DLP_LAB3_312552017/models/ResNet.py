import torch.nn as nn
import torch
from tqdm import tqdm

class ResNetBasicBlock(nn.Module):
    '''
    ResNet Paper, Figure 3, Right
    '''
    def __init__(self, in_channels, out_channels, changing_channels) -> None:
        '''
        changing_channels means we want to change the output channel of cnn layers
        i.e., the dotted line in figure 3
        '''
        super().__init__()

        if changing_channels:
            stride = 2
        else:
            stride = 1
            
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
          
 
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
         
        )        

        
        if changing_channels:
            # ResNet Paper, 
            # 3.3. Network Architectures, Residual Network option (B)
            # 4.1. ImageNet Classification, Identity vs. Projection Shortcuts
            self.shortcut  = nn.Sequential(
                # when the shortcuts go across feature maps of two
                # sizes, they are performed with a stride of 2
                nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=2, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
               
            )
        else:
            # ResNet Paper, Residual Network option (A)
            # 4.1. ImageNet Classification, Residual Networks.
            self.shortcut  = nn.Sequential()
            
        self.res_out = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # ResNet Paper, Figure 5
        input = x
        x = self.conv_block1(x)
        x = self.conv_block2(x) 
        input = self.shortcut(input)
        x = self.res_out(x + input)
        return x



class ResNetBottleneck(nn.Module):
    '''
    ResNet Paper, Figure 5, Right
    '''
    def __init__(self, in_channels, channels, out_channels, changing_channels, strides=2) -> None:
        super().__init__()
     
        if changing_channels:
            stride = 2
        else:
            stride = 1

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=(1,1), stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

        )


        self.conv_block2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3,3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=(1,1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),

        )

        if changing_channels:
            # ResNet Paper, Residual Network option (B)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=2, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),

            )
        else:
            # ResNet Paper, Residual Network option (A)
            self.shortcut = nn.Sequential()

        self.res_out = nn.ReLU(inplace=True)


    def forward(self, x):
        input = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        input = self.shortcut(input)
        x = self.res_out(x + input)
        return x
    
class ResNet(nn.Module):
    '''
    ResNet Paper, Figure 3, Right
    '''
    def __init__(self, architectures, stacking_num, labels_num) -> None:
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7,7), stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), 
     
        )

        self.conv_block2_max_pool = nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=1)

        if architectures == 'basic':
            # the last argument means
            self.conv_block2 = self.basic_blocks_building(64, 64, stacking_num[0], False)
            self.conv_block3 = self.basic_blocks_building(64, 128, stacking_num[1], True)
            self.conv_block4 = self.basic_blocks_building(128, 256, stacking_num[2], True)
            self.conv_block5 = self.basic_blocks_building(256, 512, stacking_num[3], True)

            self.fc = nn.Linear(512, labels_num)

        elif architectures == 'bottleneck':
            self.conv_block2 = self.bottleneck_blocks_building(64, 64, 256, stacking_num[0], True)
            self.conv_block3 = self.bottleneck_blocks_building(256, 128, 512, stacking_num[1], True)
            self.conv_block4 = self.bottleneck_blocks_building(512, 256, 1024, stacking_num[2], True)
            self.conv_block5 = self.bottleneck_blocks_building(1024, 512, 2048, stacking_num[3], True)
 
            self.fc = nn.Linear(2048, labels_num)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))


    def basic_blocks_building(self, in_channels, out_channels, stacking_num, changing_channels) -> nn.Sequential:
        '''
        building one of layers, e.g., conv2_x or conv3_x in Table 1,
        except for the conv2_x, the others need to change the channels
        to do the shortcut at the first block in the layer
        '''
        layers = []
        for i in range(stacking_num): 
            if i == 0 and changing_channels == True: # each layer, first stack
                layers.append(ResNetBasicBlock(in_channels, out_channels, True))
            else:
                layers.append(ResNetBasicBlock(in_channels, out_channels, False))
            in_channels = out_channels
        return nn.Sequential(*layers)
        
    def bottleneck_blocks_building(self, in_channels, channels, out_channels, stacking_num, changing_channels) -> nn.Sequential:
        '''
        in the table 1, we can first see the current input channel is the previous output channel,
        the middle channel is half size of output channel,
        and the output channel is dobble size of the previous output channel
        '''
        layers = []
        for i in range(stacking_num):
            if i == 0 and changing_channels == True:
                layers.append(ResNetBottleneck(in_channels, channels, out_channels, True))
            else:
                layers.append(ResNetBottleneck(in_channels, channels, out_channels, False))
            in_channels = out_channels

        
        return nn.Sequential(*layers)
        
         
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2_max_pool(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
 


def basic_blocks_building(in_channels, out_channels, stacking_num, changing_channels) -> nn.Sequential:
    '''
    building one of layers, e.g., conv2_x or conv3_x in Table 1,
    except for the conv2_x, the others need to change the channels
    to do the shortcut at the first block in the layer
    '''
    layers = []
    for i in range(stacking_num): 
        if i == 0 and changing_channels == True: # each layer, first stack
            layers.append(ResNetBasicBlock(in_channels, out_channels, True))
        else:
            layers.append(ResNetBasicBlock(in_channels, out_channels, False))
        in_channels = out_channels
    return nn.Sequential(*layers)


def bottleneck_blocks_building(in_channels, channels, out_channels, stacking_num, changing_channels) -> nn.Sequential:
    '''
    in the table 1, we can first see the current input channel is the previous output channel,
    the middle channel is half size of output channel,
    and the output channel is dobble size of the previous output channel
    '''
    layers = []
    for i in range(stacking_num):
        if i == 0 and changing_channels == True:
            layers.append(ResNetBottleneck(in_channels, channels, out_channels, True))
        else:
            layers.append(ResNetBottleneck(in_channels, channels, out_channels, False))
        in_channels = out_channels

    
    return nn.Sequential(*layers)

# basic = basic_blocks_building(64, 128, 2, True)
# print(basic)

# bottle = bottleneck_blocks_building(256, 128, 512, 4, True)
# print(bottle)