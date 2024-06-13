import torch.nn as nn   
import torch
import os
def activate(fn):
    if fn == 'elu':
        return nn.ELU()
    elif fn == 'relu':
        return nn.ReLU()
    elif fn == 'leakyRelu':
        return nn.LeakyReLU()
    
class SupervisedLearning():
    def __init__(self) -> None:
        pass
 
    @staticmethod
    def fit(model, train_loader, train_acc, train_loss, optimizer, loss_fn):
        model.train()


        tol_loss = 0
        tol_acc = 0 
        for batch, (X, y) in enumerate(train_loader):
            #print(f"batch size: {batch}")

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

        #print(f"tol data len an epoch: {len(train_loader.dataset)}") #1080
        # acc and loss
        train_acc.append(tol_acc / len(train_loader.dataset) * 100)
        train_loss.append(tol_loss / len(train_loader.dataset))
 
        return train_acc, train_loss
    
    @staticmethod
    def predict(model, test_loader, test_acc, test_loss, loss_fn, eval=""):
        model.eval()

        tol_acc = 0
        tol_loss = 0 

        with torch.no_grad():
            for batch, (X, y) in enumerate(test_loader):
                
                if eval == "demo":
                    X = X.cpu()  
                    y = y.cpu()
                pred = model(X)

                # loss
                loss = loss_fn(pred, y)
                tol_loss += loss.item()

                # test accuracy
                output = torch.argmax(pred, 1)
                acc = (output == y).sum().item()
                tol_acc += acc
            test_acc.append( tol_acc / len(test_loader.dataset) * 100)
            test_loss.append(tol_loss / len(test_loader.dataset))


            return test_acc, test_loss
        
    @staticmethod
    def evaluate(model, test_dataloader) -> list:
        checkpoint = os.getcwd() + "\\best_model\\" + 'best_model' + '.pth'
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
 
        test_acc = []
        highest_acc_eval = 0
        for epoch in range(1, 10+1):
            test_acc, _ = SupervisedLearning.predict(model, test_dataloader, test_acc, [], model.loss, "demo")
            
            if test_acc[-1] > highest_acc_eval:
                highest_acc_eval = test_acc[-1]
        print(f"Highest Testing Accuracy: {highest_acc_eval:.2f}%")
        print(f"Model: {checkpoint['best_model']}")
        print(f"Activation Function: {checkpoint['best_act_fn']}")