import os
import shutil
import pandas as pd 
from PIL import ImageFile, Image
import math
from torchvision import transforms
import random
from torchvision.transforms import functional as F


def transformer() -> transforms.Compose:
    # transform
    transform = transforms.Compose([

        transforms.RandomHorizontalFlip(),  # ResNet Paper 3.4. Implementation, Randomly flip the image horizontally
        transforms.RandomVerticalFlip(),  
        transforms.RandomRotation(degrees=(-90, 90)),  

    ])
    return transform

image_size = 512
def crop_img(img):

    # Downsample the image to a fixed resolution of 256x256
    # First, rescale the image such that the shorter side is of length 256
    shorter_side = min(img.size)
    scale_factor = image_size / shorter_side
    new_size = (round(img.size[0] * scale_factor), round(img.size[1] * scale_factor))
    img_rescaled = img.resize(new_size, Image.BICUBIC)
        
    # Then, crop out the central 256x256 patch from the resulting image
    left_margin = (img_rescaled.width - image_size) / 2
    top_margin  = (img_rescaled.height - image_size) / 2
    img_final = img_rescaled.crop((left_margin, top_margin, left_margin + image_size, top_margin + image_size))

    # Take a 224x224 pixels part of the image
    left_margin = (img_final.width - 224) / 2
    top_margin = (img_final.height - 224) / 2
    img_final = img_final.crop((left_margin, top_margin, left_margin + 224, top_margin + 224))

    return img_final




def gen_train_dataset():

    df = pd.read_csv(train_imgs)

    # To get zero_label images, transform them, and save them
    one_label_count = df[df['label'] == 1].shape[0]
    zero_label_count = df[df['label'] == 0].shape[0]
    balance_num = abs(one_label_count - zero_label_count)
    # print(balance_num)

    last_img_path = df['Path'].iloc[-1]
    last_img_num = last_img_path.split('/')[-1].split('.')[0] # 7994
    last_img_num = int(last_img_num)
    last_img_nums = last_img_num
    print(last_img_nums)


    # use transformer to do data argumentation
    transforms = transformer()
    zero_label_imgs_path = df[df['label'] == 0]['Path']
    img_paths = df['Path'] 


    img_num = -1
    my_data_list = []
    for img in df.itertuples(): 
        '''
        Copy original dataset to another file
        '''
        img_path = img.Path  
        img_label = img.label

        # open the image
        path = (img_root + img_path)
        img = Image.open(path)
        print(f"open img: {path}, label: {img_label}")

        # maybe crop images here
        img_final = crop_img(img)

        # save the img
        img_num = img_num + 1
        save_path = dataset_paths[0] + "\\" + str(img_num) + '.bmp'
        img_final.save(save_path)
        save_path_relative = './lab3-train/' + str(img_num) + '.bmp'
        # create new list
        new_data = {'Path': save_path_relative, 'label': img_label}   
        my_data_list.append(new_data)

        print(f"save img: {save_path}")
        print('\n')

    for img_path in zero_label_imgs_path:
        '''
        create more zero label imgs for balancing the dataset
        '''
        # open the image
        path = (img_root + img_path)
        img = Image.open(path)
        print(f"open img {path}, label: 0")
        
        # transform, and cropping
        img_final = transforms(img)
        img_final = crop_img(img)

        # Save the image
        last_img_num = last_img_num + 1
        save_path = dataset_paths[0] + "\\" + str(last_img_num) + '.bmp'
        img_final.save(save_path)
        print(f"save img: {save_path}")
        print('\n')

        save_path_relative = './lab3-train/' + str(last_img_num) + '.bmp'
        # create new list
        new_data = {'Path': save_path_relative, 'label': 0}   
        my_data_list.append(new_data)

    extra_data_list = []
    last_img_num = last_img_num + 1
    for i, data in enumerate(my_data_list):
        '''
        use transformer to double the dataset
        '''
        img_path = data['Path']
        img_label = data['label']  # This is the original label

        # Open the image
        img = Image.open('D:' + img_path)
        print(f"open img {img_path}, label: {img_label}")

        # apply the transformation
        transformed_img = transforms(img)

        # Define the new filename
        new_filename = f"{last_img_num + i}.bmp"

        # save the transformed image with the new filename
        save_path = dataset_paths[0] + "\\" + new_filename

        transformed_img.save(save_path)
        print(f"save img: {save_path}")
        print('\n')

        # update the path and label in my_data_list
        save_path_relative = './lab3-train/' + new_filename
        new_data = {'Path': save_path_relative, 'label': img_label}   
        extra_data_list.append(new_data)


    df_my_data = pd.DataFrame(my_data_list)
    df_extra_data = pd.DataFrame(extra_data_list)
    pd.concat([df_my_data, df_extra_data]).to_csv('./database/extra_extend_train.csv', index=False)


def gen_valid_dataset():
    my_data_list = []
    df = pd.read_csv(valid_imgs)
    transforms = transformer()
 

    for img in df.itertuples(): 
        '''
        Copy, and Crop original dataset to another file
        '''
        img_path = img.Path  
        img_label = img.label
        img = Image.open(img_root + img_path)

        print(f"open img: {img_path}, label: {img_label}")

        transformed_img = transforms(img)
        # transformed_img = crop_img(img)

        save_path = dataset_paths[1] + "\\" + str(img_path.split('/')[-1])

        transformed_img.save(save_path)
        print(f"save img: {save_path}")
        print('\n')
        save_path_relative = './lab3-valid/' + str(img_path.split('/')[-1])
        new_data = {'Path': save_path_relative, 'label': img_label}   

        my_data_list.append(new_data)
        df_my_data = pd.DataFrame(my_data_list)
        #df_my_data.to_csv('./database/extend_valid.csv', index=False)


def gen_test_dataset():

    df = pd.read_csv(test_imgs)


    for img in df.itertuples(): 
        '''
        Copy, and Crop original dataset to another file
        '''
        img_path = img.Path  
  
        img = Image.open(img_root + img_path)

        print(f"open img: {img_path}")


        cropped_img = crop_img(img)

        save_path = dataset_paths[2] + "\\" + str(img_path.split('/')[-1])

        cropped_img.save(save_path)
        print(f"save img: {save_path}")
        print('\n')



img_root = 'D:'
dataset_paths = ['D:\\lab3-train', 'D:\\lab3-valid', 'D:\\lab3-test']
train_imgs = './database/train.csv'
valid_imgs = './database/valid.csv'
test_imgs = './kaggle/resnet_18_test.csv'
for paths in dataset_paths:
    if not os.path.exists(paths):
         os.makedirs(paths)
gen_train_dataset()
gen_test_dataset()
gen_valid_dataset()
