
from torch.utils.data import DataLoader
from dataloader import IclevrLoader
import argparse
import torch.nn as nn
from model import ClassConditionedUnet
 
import torch
 

from ddpm import train_step, val_step
from diffusers.optimization import get_cosine_schedule_with_warmup
import os

torch.cuda.empty_cache()


# hyper parameters
parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--batch", default=4, type=int)
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument('--data_fname', default="iclevr", type=str)
parser.add_argument('--data_dir', default="../Lab6_Dataset", type=str)
parser.add_argument('--save_root', default="./result", type=str, help="The path to save your data")
parser.add_argument('--ckpt_path', default="./result/final_model.pth", type=str, help="The path to save your weight")
parser.add_argument('--test', action='store_true')
args = parser.parse_args()
 

# load data
train_data =  IclevrLoader(args.data_dir, args.data_fname, 'train')
train_dataloader = DataLoader(train_data, batch_size=args.batch, shuffle=True)

test_dataset = IclevrLoader(args.data_dir, args.data_fname, 'test')
test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

new_test_dataset = IclevrLoader(args.data_dir, args.data_fname, 'new_test')
new_test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)


# use gpu
device_id = torch.cuda.current_device()
print( f"device id: {device_id}, device name: {torch.cuda.get_device_name(device_id)}" )
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f'Using device: {torch.cuda.get_device_name(device)}')

# create model
model = ClassConditionedUnet().to(device)

# training technique
loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=200,
    num_training_steps=len(train_dataloader) * args.epochs,
)

args.loss_fn = loss_fn
args.optimizer = optimizer
args.scheduler = scheduler

# just testing
if args.test == True:
    checkpoint = torch.load(args.ckpt_path)
    new_test_acc_list = checkpoint['new_test_acc_list'][:200]
    test_acc_list = checkpoint ['test_acc_list'][:200]

else:
    # start training
    os.makedirs(args.save_root, exist_ok=True)
    train_loss_list = []
    test_acc_list = []
    new_test_acc_list = []
    best_test_acc = 0
    best_new_test_acc = 0
    for epoch in range(1, args.epochs+1):
    
        train_loss_list   = train_step(train_dataloader, model, args, train_loss_list, epoch)
        test_acc_list     = val_step(test_dataloader, model, args, test_acc_list, epoch,  'test')
        new_test_acc_list = val_step(new_test_dataloader, model, args, new_test_acc_list, epoch, 'new_test')
    

    
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"Test Accuracy: {test_acc_list[-1] * 100:.2f}%, New Test Accuracy: {new_test_acc_list[-1]* 100:.2f}%, Loss: {train_loss_list[-1]:.5f}")

        with open('{}/train_record.txt'.format(args.save_root), 'a') as train_record:
            train_record.write(('[Epoch: %02d] loss: %.5f | test acc: %.5f | new_test acc: %.5f\n' % (epoch, train_loss_list[-1], test_acc_list[-1], new_test_acc_list[-1])))

        if test_acc_list[-1] >= best_test_acc and new_test_acc_list[-1] >= best_new_test_acc:
            best_test_acc = test_acc_list[-1] 
            best_new_test_acc = new_test_acc_list[-1]
            print(f"----------update model---------")
            torch.save({    
                'model_state_dict': model.state_dict(), 
                'test_acc': best_test_acc,
                'new_test_acc': best_new_test_acc,
                'epochs': epoch
            }, os.path.join(args.save_root, "best_model.pth"))
            print(f"----------model saved----------")

        if epoch == args.epochs:
            torch.save({    
                'model_state_dict': model.state_dict(), 
                'test_acc_list': test_acc_list,
                'new_test_acc_list': new_test_acc_list,
                'train_loss_list': train_loss_list,
                'epochs': epoch
            }, os.path.join(args.save_root, "final_model.pth"))
            print(f"----------final model saved----------")


from utils import Result
fig_path = os.path.join(args.save_root, "trainingCurve.png")
Result.plot_result(test_acc_list, new_test_acc_list, fig_path)