import pandas as pd

from models.ResNet import ResNet 
from torch.utils.data import DataLoader
from dataloader import LeukemiaLoader
import torch 
from utils import *
import shutil
import argparse
import sys

your_student_id = '312552017'
torch.cuda.empty_cache()


    
def ResNet18(labels_num) -> ResNet:
    return ResNet('basic', [2,2,2,2], labels_num)

def ResNet50(labels_num) -> ResNet:
    return ResNet('bottleneck', [3,4,6,3], labels_num)

def ResNet152(labels_num) -> ResNet:
    return ResNet('bottleneck', [3,8,36,3], labels_num)


def save_result(csv_path, predict_result, model_csv_name):
    df = pd.read_csv(csv_path)
    new_df = pd.DataFrame()
    new_df['ID'] = df['Path']
    new_df["label"] = predict_result
    new_df.to_csv(f"./{your_student_id}_{model_csv_name}.csv", index=False)

def model_select(model_name, labels_num=2):
    if model_name == 'ResNet18':
        model = ResNet18(labels_num).to(device)
    elif model_name == 'ResNet50':
        model = ResNet50(labels_num).to(device)
    elif model_name == 'ResNet152':
        model = ResNet152(labels_num).to(device)

    else:
        raise ValueError('No such model')
    return model

def show_results(train_acc_dict, test_acc_dict):
    plt.figure(figsize=(10, 5))
    plt.title(f'ResNet Accuracy Curve', fontsize=18)
    
    for label, acc in train_acc_dict.items():
        plt.plot(acc, label=label + '_Train')
    for label, acc in test_acc_dict.items():
        plt.plot(acc, label=label + '_Valid')
        max_point = np.argmax(acc)
        plt.annotate(f'{np.max(acc):.2f}', (max_point, acc[max_point]), textcoords="offset points", xytext=(-10,10), ha='center', color='orange')

    plt.xlabel('Epoch', fontsize=14)   
    plt.ylabel('Accuracy (%)', fontsize=14)

    plt.yticks(range(50, 100 + 1, 5))

    plt.legend(loc='lower right')
    plt.savefig( 'ResNet_Result_Comparison'  + '.png')
    plt.show()


# get gpu device
device_id = torch.cuda.current_device()
print( f"device id: {device_id}, device name: {torch.cuda.get_device_name(device_id)}" )
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f'Using device: {torch.cuda.get_device_name(device)}')

if __name__ == "__main__":
    num = 50

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', help='train, unit_test, demo, test')
    parser.add_argument('--model', default=f'ResNet{num}', help='ResNet18, ResNet50, ResNet152')
    parser.add_argument('--exp_result', type=bool, default=False, help='Plot Comparison figures')
    args = parser.parse_args()


    model = model_select(args.model)

    # setting
    labels_num = 2
    model_name = args.model
    model_csv_name = f'resnet{num}'
    

    labels = [0,1]

    # hyperparameters setting
    batch_size = 16#64
    epochs = 40
    learning_rate = 0.001#0.003

    
  
    # preprocessing 
    root = os.getcwd()
    print(f"root path: {root}")
    dataset_path = 'D:'

    if args.mode != 'demo':
        train_data = LeukemiaLoader(dataset_path, 'train', num)
        extend_valid_data = LeukemiaLoader(dataset_path, 'extend_valid', num)
    valid_data = LeukemiaLoader(dataset_path, 'valid', num)
    test_data  = LeukemiaLoader(dataset_path, 'test', num)

    if args.mode != 'demo':
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        extend_valid_dataloader =  DataLoader(extend_valid_data, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_dataloader  = DataLoader(test_data, batch_size=batch_size, shuffle=False)
 


    if args.mode == 'demo':

        # os.chdir(f'{model_name}')

        model_names = ['ResNet18', 'ResNet50', 'ResNet152']
        for names in model_names:
            model = model_select(names)
           
            print(f"{names} inference on validation data")

            os.chdir(names)
      
            
            y_true, y_pred = SupervisedLearning.evaluate(model, names, valid_dataloader, 'valid')
        
            os.chdir(root)

        
            differences = [i for i, j in zip(y_true, y_pred) if i != j]
            num_differences = len(differences)
            print(f"{names} Highest Accuracy {(len(y_pred) - num_differences)/len(y_pred):.2f}")
            print(f"--------------------------------------------------------------------------")

         
        sys.exit()

    if args.exp_result == True:
        train_acc_dict = {}
        valid_acc_dict = {}
        
        model_names = ['ResNet18', 'ResNet50', 'ResNet152']
        for names in model_names:
            model = model_select(names)
           
            print(f"{names}")

            os.chdir(names)
            train_acc, valid_acc = SupervisedLearning.evaluate(model, names, valid_dataloader, 'valid', True)

            train_acc_dict[names] = train_acc
            valid_acc_dict[names] = valid_acc
            
            y_true, y_pred = SupervisedLearning.evaluate(model, names, valid_dataloader, 'valid')
            Result.plot_confusion_matrix(y_true, y_pred, labels, names)
            os.chdir(root)
        show_results(train_acc_dict, valid_acc_dict)

            
        sys.exit()
    if args.mode == 'test':

        root = os.getcwd()
        os.chdir(f'result_{model_name}')
        y_pred = SupervisedLearning.evaluate(model, model_name, test_dataloader, 'test')
        print(f"{len(y_pred)}")
        os.chdir(root)
        df = pd.read_csv(f'./kaggle/resnet_{str(num)}_test.csv')

        if len(y_pred) != len(df):
            print(f"The length of the list ({len(y_pred)}) doesn't match  ({len(df)}).")
        else:
            #df['label'] = y_pred

            save_result(f'./kaggle/resnet_{str(num)}_test.csv', y_pred, model_csv_name)
        sys.exit()
    # create model, traning by using gpu
 
    best_model_save_path = model_name + '.pt' # model save path
    last_model_save_path = model_name + '_last_epoch' + '.pt' # last epoch save path

    # training techniques
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    loss_fn = torch.nn.CrossEntropyLoss()


    # training and testing
    train_acc = []
    train_loss = []
    valid_acc = []
    valid_loss = []
    ex_valid_acc = []
    ex_valid_loss = []
    best_acc = 0

    
    for epoch in range(1, epochs+1):

        print(f"Epoch {epoch}/{epochs}")
        train_acc, train_loss = SupervisedLearning.fit( 
                                                       model, 
                                                       train_dataloader, 
                                                       train_acc, 
                                                       train_loss,
                                                       optimizer,
                                                       loss_fn
                                                    )
        print(f"train_loss: {train_loss[-1]}, train_acc: {train_acc[-1]:.2f}%")

        ex_valid_acc, ex_valid_loss = SupervisedLearning.fit( 
                                                       model, 
                                                       extend_valid_dataloader, 
                                                       ex_valid_acc, 
                                                       ex_valid_loss,
                                                       optimizer,
                                                       loss_fn
                                                    )
        print(f"ex_valid_loss: {ex_valid_loss[-1]}, ex_valid_acc: {ex_valid_acc[-1]:.2f}%")

        valid_acc, valid_loss = SupervisedLearning.predict(
                                                        model,
                                                        valid_dataloader,
                                                        valid_acc, 
                                                        valid_loss,
                                                        loss_fn
                                                    )
        print(f"valid_loss: {valid_loss[-1]}, valid_acc: {valid_acc[-1]:.2f}%")
        print(f"-------------------------------")

 
        Isupdate = False
        if valid_acc[-1] > best_acc:
            best_acc = valid_acc[-1]
            print(f"best_acc: {best_acc:.2f}")
            Isupdate = True
    
        # update the model, compared to the previous model
        
        curr_path = os.getcwd()
        result_path = curr_path  + '/result_' + model_name


        if os.path.exists(result_path):
         
            checkpoint = result_path + "/" + model_name + '.pt'

  
            checkpoint = torch.load(checkpoint)
     
            if best_acc > checkpoint['best_acc']:
 
                shutil.rmtree(result_path)
                Isupdate = True
            else:
                Isupdate = False
  
        else:
            Isupdate = True

        if Isupdate:
            print(f"----------update model---------")

            if not os.path.exists(result_path):
                os.makedirs(result_path) 
            os.chdir(result_path)
            torch.save({ 
                'model_state_dict': model.state_dict(), 
                'best_acc': best_acc,
                'valid_loss': valid_loss,
                'valid_acc': valid_acc,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'epochs': epoch
            }, best_model_save_path)
            print(f"----------model saved----------")
            os.chdir(curr_path)
        if epoch == epochs:
            os.chdir(result_path)
            torch.save({ 
                'model_state_dict': model.state_dict(), 
                'best_acc': best_acc,
                'valid_loss': valid_loss,
                'valid_acc': valid_acc,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'epochs': epoch
            }, last_model_save_path)

    # show the final result
    Result.show_acc_result(epoch, train_acc, valid_acc, model_name)
    Result.show_loss_result(epoch, train_loss, valid_loss, model_name)

    y_true, y_pred = SupervisedLearning.evaluate(model, model_name, valid_dataloader, 'valid')
    
    Result.plot_confusion_matrix(y_true, y_pred, labels, model_name)

    # run testing for kaggle
    #  y_pred = SupervisedLearning.evaluate(model, model_name, test_dataloader, 'test')
