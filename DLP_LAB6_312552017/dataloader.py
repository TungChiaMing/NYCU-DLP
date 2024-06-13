import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms



def transformer() -> transforms.Compose:
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
 
    return transform


 

def getData(root, mode):
    path = os.path.join(root, mode + '.json')
    data_json = json.load(open(path))


    if mode == 'train':
        image_all = list(data_json.keys())
        text_all = list(data_json.values())
        return image_all, text_all
    else:
        return data_json
    
class IclevrLoader(Dataset):
    def __init__(self, root, data_fname, mode):
        """
        Args:
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.data_fname = data_fname
        self.mode = mode

        if mode == 'train':
            self.img_name, self.label = getData(root, mode)
            self.transform = transformer()
        else:
            self.label = getData(root, mode)
            self.transform = None
 
        print("> Found %d images..." % (len(self.label)))  
 
    def __len__(self):
        """'return the size of dataset"""
        return len(self.label)

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
         
        if self.mode == 'train':
            img_path = os.path.join(self.root , self.data_fname , self.img_name[index])

    
            img = Image.open(img_path).convert("RGB") # if no convert. RuntimeError: The size of tensor a (4) must match the size of tensor b (3) at non-singleton dimension 0
    
       
        label = self.label[index]
   
        label_one_hot = self.one_hot_transfrom(label)
 

        # Step 3: Transform the image
        if self.transform is not None:
            img = self.transform(img)
            return img, label_one_hot
        else:
            return label_one_hot
    
    def one_hot_transfrom(self, label):
        path = os.path.join(self.root ,'objects.json')
        one_hot_dict = json.load(open(path))
        onehot_label = np.zeros(len(one_hot_dict), dtype=np.float32)

        # if no , dtype=np.float32. 
        #    class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
        #    return forward_call(*args, **kwargs)
        #    return F.linear(input, self.weight, self.bias)
        # RuntimeError: mat1 and mat2 must have the same dtype
     
        for text in label:
            number = one_hot_dict[text]
            onehot_label[number] = 1 

        return onehot_label
