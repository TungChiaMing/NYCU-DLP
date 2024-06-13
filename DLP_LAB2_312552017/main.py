import torch     
from model.EEGNet import EEGNet
from model.DeepConvNet import DeepConvNet
import database.dataloader as dataloader
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from model.utils import *
from torch.optim.lr_scheduler import StepLR

import argparse

import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table
import sys

class BCIDataset(Dataset):
    def __init__(self, data, lable, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        
        self.data = data
        self.lable = lable
 

    def __len__(self) -> int:
        return len(self.lable)
    
    def __getitem__(self, idx) -> torch.Tensor:
        # print(type((self.lable[idx]))) # 'numpy.float64' 
         
        data = torch.from_numpy(self.data[idx])

        # make sure the consistency of
        # data type between input and the model's weights 
        data = data.float().cuda()    


        lable = torch.from_numpy(np.array(self.lable[idx]))
        # make sure the consistency of
        # data type between predicted labels and expected data type by loss function 
        lable = lable.long().cuda()
 
        if self.transform:
            pass
        elif self.target_transform: 
            pass
        return data, lable
 

def show_results(train_acc_dict, test_acc_dict, model_name):
    plt.figure(figsize=(10, 5))
    plt.title(f'Activation function comparison ({model_name})', fontsize=18)
    
    for label, acc in train_acc_dict.items():
        plt.plot(acc, label=label + '_train')
    for label, acc in test_acc_dict.items():
        plt.plot(acc, label=label + '_test')
        max_point = np.argmax(acc)
        plt.annotate(f'{np.max(acc):.2f}', (max_point, acc[max_point]), textcoords="offset points", xytext=(-10,10), ha='center', color='orange')

    plt.xlabel('Epoch', fontsize=14)   
    plt.ylabel('Accuracy (%)', fontsize=14)

    plt.yticks(range(70, 100 + 1, 5))

    plt.legend(loc='lower right')
    plt.savefig( model_name + '_acc' + '.png')
    plt.show()

def update_model(model, model_info, model_save_path):

    # update the model
    Isupdate = False
    result_path = os.getcwd() + '\\best_model\\' + model_save_path 
    if os.path.exists(result_path):
        checkpoint = model_save_path

        root = os.getcwd() 
        os.chdir(root + '\\best_model')
        checkpoint = torch.load(checkpoint)
        os.chdir(root)

        record_test_acc = checkpoint['test_acc']

        if model_info['best_acc']  > record_test_acc:
             
            Isupdate = True
    else:
        Isupdate = True

    if Isupdate:
        print(f"----------update model-----------")
 
        torch.save({ 
            'model_state_dict': model.state_dict(), 
            'test_acc': model_info['best_acc'],
            'best_model': model_info['best_model'],
            'best_act_fn': model_info['best_act_fn'],
            'best_epoch': model_info['best_epoch'],
        }, result_path)
        print(f"----------model saved------------")

def acc_table(df):
    df = df.round(2)
    df = df.applymap(lambda x: f'{x}%')

    fig, ax = plt.subplots(figsize=(15, 8)) # Increase size values
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    tabla = table(ax, df, loc='center', cellLoc = 'center')
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(14)  # Increase the fontsize
    tabla.scale(1.4, 1.4)  # Increase the scaling factor
    plt.tight_layout()  # Adjust layout
    plt.savefig('acc_table.png')


if __name__ == "__main__":
    # demo cmd
    # python ./main.py --mode demo
    ############
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', help='train, unit_test, demo')
    args = parser.parse_args()


    # hyperparameters setting
    batch_size = 64
    epochs = 300
    learning_rate = 1e-3
 

    # preprocessing
    path = os.getcwd()
 
    
    os.chdir(path + '\\database')
    train_data, train_label, test_data, test_label  = dataloader.read_bci_data()
    os.chdir(path) 
    #print(path)
    train_data = BCIDataset(train_data, train_label)
    test_data = BCIDataset(test_data, test_label)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    if args.mode == 'demo':
        model = EEGNet('leakyRelu')
        Optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model.compile(
            optimizer=Optimizer,
            loss=torch.nn.CrossEntropyLoss(),
             
        )
        SupervisedLearning.evaluate(model, test_dataloader)

        sys.exit()
 
    # create model, traning by using gpu
    device_id = torch.cuda.current_device()
    print( f"device id: {device_id}, device name: {torch.cuda.get_device_name(device_id)}" )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {torch.cuda.get_device_name(device)}')
    
    if args.mode == 'unit_test':
        # unit test
        model = EEGNet('leakyRelu').to(device)
        Optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model.compile(
            optimizer=Optimizer,
            loss=torch.nn.CrossEntropyLoss(),
             
        )
        # traning & testing
        training_acc = []
        training_loss = []
        testing_acc = []
        testing_loss = []
        for epoch in range(1, epochs+1):
            # traning 
            training_acc, training_loss = SupervisedLearning.fit(model, train_dataloader, training_acc, training_loss, model.optimizer, model.loss)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}")
                print(f"Training Accuracy: {training_acc[-1]:.2f}%, Loss: {training_loss[-1]}")
            # testing
            testing_acc, testing_loss = SupervisedLearning.predict(model, test_dataloader, testing_acc, testing_loss, model.loss)
            if epoch % 10 == 0:
                print(f"Testing  Accuracy: {testing_acc[-1]:.2f}%")
                print(f"-------------------------------")

        sys.exit()


    # diff results
    model_list = []
    model_list.append('EEGNet')
    model_list.append('DeepConvNet')
   
    act_fn_list = []
    act_fn_list.append('leakyRelu')
    act_fn_list.append('elu')
    act_fn_list.append('relu')
    df = pd.DataFrame.from_dict({model: {act_fn: 0.0 for act_fn in act_fn_list} for model in model_list}, orient='index')


    # highest accuracy
    highest_acc = 0
    model_save_path = 'best_model.pth'
    for model_name in model_list:
        
 
        train_acc_dict = {}
        test_acc_dict = {}


        for act_fn in act_fn_list:
            # highest acc each one
            highest_acc_all = {}
            highest_acc_inst = 0

            # model info
            model_info = {}
            model_info['model_name'] = model_name
            model_info['act_fn'] = act_fn
            model_info['highest_acc'] = highest_acc
            
            # model instance
            inst = model_name + '_' + act_fn
            if inst not in highest_acc_all:
                highest_acc_all[inst] = 0 

            # create a new model with diff fn
            if model_name == 'EEGNet':
                
                model = EEGNet(act_fn).to(device)
            elif model_name == 'DeepConvNet':
                
                model = DeepConvNet(act_fn).to(device)
                
            # traning techniques
            Optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            model.compile(
                optimizer=Optimizer,
                loss=torch.nn.CrossEntropyLoss(),
                 
            )

            # traning & testing
            training_acc = []
            training_loss = []
            testing_acc = []
            testing_loss = []
            for epoch in range(1, epochs+1):
                # traning 
                training_acc, training_loss = SupervisedLearning.fit(model, train_dataloader, training_acc, training_loss, model.optimizer, model.loss)
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}/{epochs}")
                    print(f"Training Accuracy: {training_acc[-1]:.2f}%, Loss: {training_loss[-1]}")
                # testing
                testing_acc, testing_loss = SupervisedLearning.predict(model, test_dataloader, testing_acc, testing_loss, model.loss)
                if epoch % 10 == 0:
                    print(f"Testing  Accuracy: {testing_acc[-1]:.2f}%")
                    print(f"-------------------------------")

                if testing_acc[-1] > highest_acc_inst: 
                    highest_acc_all[inst] = highest_acc_inst
                    highest_acc_inst = testing_acc[-1]

                    df.loc[model_name, act_fn] = highest_acc_inst

                if testing_acc[-1] > highest_acc:
                    highest_acc = testing_acc[-1]
                    model_info['best_model'] = model_name
                    model_info['best_act_fn'] = act_fn
                    model_info['best_epoch'] = epoch
                    model_info['best_acc'] = highest_acc

                    update_model(model, model_info, model_save_path)

            train_acc_dict[act_fn] = training_acc
            test_acc_dict[act_fn] = testing_acc
        show_results(train_acc_dict, test_acc_dict, model_name)
  
        print(f"Highest Testing Accuracy: {highest_acc:.2f}%")
        print(f"-------------------------------")
        # print(f"best model: {model_info['best_model']}")
        # print(f"best act_fn: {model_info['best_act_fn']}")
        # print(f"best epoch: {model_info['best_epoch']}")
 
    acc_table(df)



