import pandas as pd
from PIL import Image
from torch.utils import data

from torchvision import transforms
import os
import random
import torchvision.transforms.functional as F
def getData(mode, model):
    if mode == 'train':
        df = pd.read_csv('./database/extra_extend_train.csv')
        path = df['Path'].tolist()
        label = df['label'].tolist()
        return path, label

    elif mode == "extend_valid":
        df = pd.read_csv('./database/extend_valid.csv')
        path = df['Path'].tolist()
        label = df['label'].tolist()
        return path, label

    elif mode == "valid":
        df = pd.read_csv('./database/valid.csv')
        path = df['Path'].tolist()
        label = df['label'].tolist()
        return path, label
    
    else:
        df = pd.read_csv(f'./kaggle/resnet_{model}_test.csv')
        path = df['Path'].tolist()
        return path




def transformer(mode) -> transforms.Compose:

    if mode == 'train':
        # Training transformations
        transform = transforms.Compose([
            transforms.ToTensor(),  # step3, hint
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
            #transforms.RandomHorizontalFlip(),  # ResNet Paper 3.4. Implementation, Randomly flip the image horizontally
            #transforms.RandomVerticalFlip(), 
            #transforms.RandomRotation(degrees=(-90, 90)),  

        ])
    elif mode == 'valid':
        # Valid transformations
        transform = transforms.Compose([

            transforms.ToTensor(),  # step3, hint
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
        ])
    else:
        # Testing
        transform = transforms.Compose([
            transforms.ToTensor(),  # step3, hint
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
        ])
    return transform


class LeukemiaLoader(data.Dataset):
    def __init__(self, root, mode, model):
        """
        Args:
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root

        if mode == 'test':
            self.img_name = getData(mode, model)
            # print(f"Run testing on ResNet_{model}, save it to the kaggle result if runing testing")
        else:
            self.img_name, self.label = getData(mode, model)


        self.mode = mode
        self.transform = transformer(mode)
        print("> Found %d images..." % (len(self.img_name)))  

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
         
        img_path = self.root + self.img_name[index]

 
        img = Image.open(img_path)
 
        if self.mode != 'test':
            label = self.label[index]

 
        # Step 3: Transform the image
        if self.transform is not None:
            img = self.transform(img)
        else:
            raise ValueError('Transform is None')
        
        if self.mode == 'test':
            return img
        return img, label