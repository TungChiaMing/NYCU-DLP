import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import torch
from tqdm import tqdm
import os 
from models.ResNet import ResNet
from torch.utils.data import DataLoader

class Result():
    def __init__(self) -> None:
        pass
    @staticmethod
    def show_acc_result(epochs, train_acc, valid_acc, model_name):
        range = np.arange(1, epochs+1)
 
        plt.figure(figsize=(10, 5))
        plt.title(model_name, fontsize=18)
        plt.plot(range, train_acc, label='Train')
        plt.plot(range, valid_acc, label='Valid')
        plt.legend(loc='upper left')
        plt.xlabel('Epoch', fontsize=14)   
        plt.ylabel('Accuracy (%)', fontsize=14)   
        plt.savefig( model_name + '_acc' + '.png')
        plt.show()
        

    @staticmethod
    def show_loss_result(epochs, train_loss, valid_loss, model_name):
        range = np.arange(1, epochs+1)
        plt.figure(figsize=(10, 5))
        plt.title(model_name, fontsize=18)
        plt.plot(range, train_loss, label='Train')
        plt.plot(range, valid_loss, label='Valid')
        plt.legend(loc='upper left')
        plt.xlabel('Epoch', fontsize=14)   
        plt.ylabel('Loss', fontsize=14)   
        plt.savefig( model_name + '_loss' + '.png')
        plt.show()

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, labels, model_name):
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
        cm_normalized = np.round(cm / np.sum(cm, axis=1).reshape(-1, 1), decimals=2)
        sns.heatmap(cm_normalized, annot=True, cmap='Blues')
        plt.title(f'{model_name} Normalized Confusion Matrix')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.savefig( model_name + '_confusion_matrix' + '.png')
        plt.show()
        

device = "cuda:0" if torch.cuda.is_available() else "cpu"
class SupervisedLearning():
    def __init__(self) -> None:
        pass
 
    @staticmethod
    def fit(model, train_loader, train_acc, train_loss, optimizer, loss_fn):
        model.train()


        tol_loss = 0
        tol_acc = 0 
        for batch, (X, y) in enumerate(tqdm(train_loader)):
        
            # pred
            X = X.to(device)
            y = y.to(device)
            pred = model(X)

            # loss
            loss = loss_fn(pred, y)
            tol_loss += loss.item()

            # gradient descent
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accuracy
            output = torch.argmax(pred, 1)
            acc = (output == y).sum().item()
            tol_acc += acc

        # acc and loss
        train_acc.append(tol_acc / len(train_loader.dataset) * 100)
        train_loss.append(tol_loss / len(train_loader.dataset))
 
        return train_acc, train_loss
    
    @staticmethod
    def predict(model, valid_loader, valid_acc, valid_loss, loss_fn):
        model.eval()

        tol_acc = 0
        tol_loss = 0 

        with torch.no_grad():
            for batch, (X, y) in enumerate(tqdm(valid_loader)):
                # pred
                X = X.to(device)
                y = y.to(device)
                pred = model(X)

                # loss
                loss = loss_fn(pred, y)
                tol_loss += loss.item()

                # test accuracy
                output = torch.argmax(pred, 1)

                acc = (output == y).sum().item()
                tol_acc += acc
        valid_acc.append( tol_acc / len(valid_loader.dataset) * 100)
        valid_loss.append(tol_loss / len(valid_loader.dataset))


        return valid_acc, valid_loss
        

    @staticmethod
    def evaluate(model, path, dataloader, mode, get_exp_result=False) -> list:
        
        checkpoint = os.getcwd() + "/" + path + '.pt'

        if get_exp_result:
            checkpoint = os.getcwd() + "/" + path + '_last_epoch' + '.pt'
        

        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
 
        model.eval()

        y_true = []  
        y_pred = []

        if mode == 'valid':
            epochs = checkpoint['epochs']
            train_acc = checkpoint['train_acc']
            valid_acc = checkpoint['valid_acc']
            if get_exp_result:
                return train_acc, valid_acc
            train_loss = checkpoint['train_loss']
            valid_loss = checkpoint['valid_loss']
            # print(epochs)
            # Result.show_acc_result(epochs, train_acc, valid_acc, path)
            # Result.show_loss_result(epochs, train_loss, valid_loss, path)
         
            with torch.no_grad(): 
                for batch, (X, y) in enumerate(tqdm(dataloader)): 
                    X = X.to(device)
                    y = y.to(device)   
                    pred = model(X)

                    # final output
                    output = torch.argmax(pred.data, 1) 
                    # y_pred.extend(output.cpu())
                    # print((output)) # tensor([0, 0, 0, 0], device='cuda:0') <class 'torch.Tensor'>
                    y_pred.extend([output_item.item() for output_item in output.cpu()])

                    # y_true.extend(y.cpu())
                    y_true.extend([y_item.item() for y_item in y.cpu()])

            
            return y_true, y_pred
        else:
            with torch.no_grad(): 
                for batch, X in enumerate(tqdm(dataloader)): 
                    X = X.to(device)
             
                    pred = model(X)

                    # final output
                    output = torch.argmax(pred.data, 1) 
                    # y_pred.extend(output.cpu())
                    # print((output)) # tensor([0, 0, 0, 0], device='cuda:0') <class 'torch.Tensor'>
                    y_pred.extend([output_item.item() for output_item in output.cpu()])
           

            
            return  y_pred
'''
path_test =  'd:\\lab3-test'

test_data = RetinopathyLoader(path_test, 'test')

test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True)


model =  ResNet('basic', [2,2,2,2]).to(device)
model.compile(
    optimizer=torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4), 
    loss=torch.nn.CrossEntropyLoss()
)

model_name = "ResNet18"
y_true, y_pred = SupervisedLearning.evaluate(model, model_name, test_dataloader)
labels = [0,1,2,3,4]
Result.plot_confusion_matrix(y_true, y_pred, labels, 'ResNet18')
'''
