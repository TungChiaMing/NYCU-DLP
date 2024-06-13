from typing import Any
import matplotlib.pyplot as plt
from sequential import Seqential
import database
import argparse


class Result:
    def __init__(self) -> None:
        pass

    @staticmethod
    def show_result(x, y, pred_y):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Ground truth', fontsize=18)
        for i in range(x.shape[0]):
            if y[i] == 0:
                plt.plot(x[i][0], x[i][1], 'ro')
            else:
                plt.plot(x[i][0], x[i][1], 'bo')
        
        plt.subplot(1, 2, 2)
        plt.title('Predict result', fontsize=18)
        for i in range(x.shape[0]):
            if pred_y[i] == 0:
                plt.plot(x[i][0], x[i][1], 'ro')
            else:
                plt.plot(x[i][0], x[i][1], 'bo')
        plt.show()

    @staticmethod
    def show_loss(loss_list, lr, opt_name, dataset):
        plt.figure(figsize=(10, 5))
        plt.title('Learning curve of ' + dataset, fontsize=18)
        plt.plot(loss_list, label='lr= '+str(lr) + ', optimizer= ' + opt_name) 
        plt.xlabel('Epoch', fontsize=14)   
        plt.ylabel('Cross Entropy Error', fontsize=14)   
        plt.legend()
        plt.show()
    @staticmethod
    def show_losses(loss_dict, opt_name, dataset):
        '''
        loss_dict = {
            "loss1": [0.1, 0.2, 0.3, 0.4],
            "loss2": [0.2, 0.3, 0.4, 0.5]
        }
        '''
        plt.figure(figsize=(10, 5))
        plt.title('Learning curve of ' + dataset, fontsize=18)
        for label, loss_list in loss_dict.items(): 
            plt.plot(loss_list, label=label) #+ ', optimizer= ' + opt_name)
        plt.xlabel('Epoch', fontsize=14)   
        plt.ylabel('Cross Entropy Error', fontsize=14)   
        plt.legend()  # To show the labels of different curves
        plt.show()

class Model:
    def __init__(self) -> None:
        pass

    @staticmethod
    def SimpleModel(out_dim, hid_units, activation_fn, hidden_fn):
        model = Seqential()
        model.add_dense(hid_units, input_dim=2, activation=hidden_fn)
        model.add_dense(hid_units, activation=hidden_fn)
        model.add_dense(out_dim, activation=activation_fn)
        return model

if __name__ == '__main__':

    # hyperparameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--activation', default='sigmoid', help='algorithm')
    parser.add_argument('--epochs', default=5500, type=int, help='epochs')
    parser.add_argument('--optimizer', default='sgd', help='optimizer')
    parser.add_argument('--lr', default=0.04, type=float, help='learning rate')
    parser.add_argument('--dataset', default='XOR', help='dataset')
    parser.add_argument('--units', default=4, type=int, help='hidden units')
    parser.add_argument('--diff', default='none', help='different hyperparameters')
    parser.add_argument('--hid_fn', default='relu', help='hidden function')
    parser.add_argument('--demo', default='True', help='output function')
    args = parser.parse_args()
    # Demo
    #   --activation sigmoid --epochs 500 --optimizer sgd --lr 0.04 --dataset linear
    #   --activation sigmoid --epochs 5500 --optimizer sgd --lr 0.04 --dataset XOR
    ########
    # XOR 100%: 
    #   --activation sigmoid --epochs 5500 --optimizer sgd --lr 0.04
    #   --activation softmax --epochs 5000 --optimizer sgd --lr 0.07
    # XOR with linear:
    #   --activation linear --epochs 10000 --optimizer sgd --lr 0.04 --dataset XOR --diff activation --demo f
    # XOR with diff optimizer
    #   --activation sigmoid --epochs 5500 --optimizer sgd --lr 0.04 --dataset XOR --diff optimizer --demo f
    # XOR with diff hidden layer activation
    #   --epochs 5500 --dataset linear --diff hid_fn --demo f
    # XOR with diff units
    #   --epochs 5500 --optimizer sgd --lr 0.04 --dataset XOR --diff unit --demo f
    # XOR with diff output layer activation
    #   --epochs 5500 --dataset linear --diff activation --demo f
    # linear 100%:
    #   --activation sigmoid --epochs 500 --optimizer sgd --lr 0.04 --dataset linear
    # linear with diff units
    #   --activation sigmoid --epochs 500 --optimizer sgd --lr 0.04 --dataset linear --diff unit --demo f
    # linear with linear:
    #   --activation linear --epochs 10000 --optimizer sgd --lr 0.04 --dataset linear --diff activation --demo f
    # linear with diff optimizer
    #   --activation sigmoid --epochs 500 --optimizer sgd --lr 0.04 --dataset linear --diff optimizer --demo f
    # linear with diff hidden layer activation
    #   --epochs 500 --dataset linear --diff hid_fn --demo f
    # linear with diff output layer activation
    #   --epochs 500 --dataset linear --diff activation --demo f
  
    if args.activation == 'softmax':
        loss_fn = 'categorical_cross_entropy'
        pred_algorithm = 'classification'
        output_dim = 2
    elif args.activation == 'sigmoid':
        loss_fn = 'binary_cross_entropy'
        pred_algorithm = 'logistic_regression'
        output_dim = 1
    elif args.activation == 'linear':
        args.hid_fn = 'linear'
        loss_fn = 'mean_square_error'
        pred_algorithm = 'logistic_regression'
        output_dim = 1

    # preprocessing
    if args.dataset == 'XOR':
        traning_x, traning_y = database.generate_XOR_easy()  
    else:
        traning_x, traning_y = database.generate_linear()  

    # create model
    model = Model.SimpleModel(output_dim, args.units, args.activation, args.hid_fn) 

    if args.demo == 'True':
        model.compile(loss=loss_fn, optimizer=args.optimizer, lr=args.lr)

        loss_list = model.fit(traning_x, traning_y, epochs=args.epochs)
        Result.show_loss(loss_list, args.lr, args.optimizer, args.dataset)

        pred = model.predict(traning_x, traning_y, algorithm=pred_algorithm) 
        Result.show_result(traning_x, traning_y, pred)
    else:
        # Discussion - try different hyperparameters
        loss_dict = {}
        for i in range(2):

            model.compile(loss=loss_fn, optimizer=args.optimizer, lr=args.lr)
    
            # traning 
            loss_list = model.fit(traning_x, traning_y, epochs=args.epochs)

            if args.diff == 'lr':
                different_result = 'lr= ' + str(args.lr)
            elif args.diff == 'unit':
                different_result = 'units= ' + str(args.units)
            elif args.diff == 'activation':
                different_result = 'activation= ' + args.activation
            elif args.diff == 'optimizer':
                different_result = 'optimizer= ' + args.optimizer
            elif args.diff == 'hid_fn':
                different_result = 'hid_fn= ' + args.hid_fn
            
            loss_dict[different_result] = loss_list
            Result.show_loss(loss_list, args.lr, args.optimizer, args.dataset)


            # testing
            # use traning as test data may be overfitting
            pred = model.predict(traning_x, traning_y, algorithm=pred_algorithm) 
            Result.show_result(traning_x, traning_y, pred)

            if args.diff == 'lr':
                args.lr = args.lr * 10
         
            elif args.diff == 'unit':
                if i==0:
                    args.units = 10
                if i==1:
                    args.units = 100
              
            elif args.diff == 'activation':
                args.activation = 'softmax'
                loss_fn = 'categorical_cross_entropy'
                pred_algorithm = 'classification'
                output_dim = 2
                args.hid_fn = 'relu'
              
            elif args.diff == 'optimizer':
                args.optimizer = 'momentum'

            elif args.diff == 'hid_fn':
                args.hid_fn = 'tanh'

            model = Model.SimpleModel(output_dim, args.units, args.activation, args.hid_fn)
            
           
    
        Result.show_losses(loss_dict, args.optimizer, args.dataset)

     

 