import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""

    #print(f"type of imgs1: {type(imgs1)}")
    
    #print(f"type of imgs2: {type(imgs2)}")
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1

    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size  
  return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        super().__init__()
        self.betas = [] # np.ones(train_len)
        
        self.max_beta = 1.0
        
        self.cyclical_mode = args.kl_anneal_type
        self.cycle = args.kl_anneal_cycle
        self.ratio = args.kl_anneal_ratio

        # self.num_iters = args.num_epoch

        self.idx = 0
        # TODO
        # raise NotImplementedError
    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0):
        self.betas = np.ones(n_iter)
        if self.cyclical_mode != 'Cyclical':
            # print(f"using kl linear mode")
            period = n_iter#self.num_iters
            step = (self.max_beta - 0.0) / (period * self.ratio)
        

            v , i = 0 , 0
            while v <= self.max_beta and (int(i) < n_iter):
                self.betas[int(i)] = v

                v += step
                i += 1
        else:
            # print(f"using kl Cyclical mode")
            period = n_iter / self.cycle
            
            step = (self.max_beta - 0.0)/(period*self.ratio)  

            for c in range(self.cycle):

                v , i = 0 , 0
                while v <= self.max_beta and (int(i+c*period) < n_iter):
                    self.betas[int(i+c*period)] = v

                    v += step
                    i += 1

        # TODO
        # raise NotImplementedError
    def get_beta(self):

        beta = self.betas[0]
       
        if self.betas.size == 1:
            return beta

        self.betas = np.delete(self.betas, 0)
        return beta
        # TODO
        # raise NotImplementedError

    def update(self):
        self.idx += 1
        # TODO
        # raise NotImplementedError

        
class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        # hyper parameters setting
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5], gamma=0.09)#BEST: milestones=[2, 5], gamma=0.1
        self.kl_annealing = kl_annealing(args, current_epoch=0) # init later

        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0

        self.training_loss = []
        self.training_mse_loss = []
        self.training_kld_loss = []
        self.training_tfr = []
        self.training_kl_weight = []
        self.ave_psnr_list = []
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        
        # result
        self.best_psnr = 0
        
    def forward(self, p_t, xt_1_hat, x_t):
        pose = self.label_transformation(p_t) # p_t is the current label 
        frame_last = self.frame_transformation(xt_1_hat) # xt_1_hat is last prediction, if t-1 == 1, xt_1 is x_1
        frame = self.frame_transformation(x_t) # x_t is the current img 
        

        # encoder
        z, mu, logvar = self.Gaussian_Predictor(frame, pose) # posterior predict

        # decoder
        d = self.Decoder_Fusion(frame_last, pose, z)
        x_t_hat = self.Generator(d)

        # loss
        mse = self.mse_criterion(x_t, x_t_hat)
        kld = kl_criterion(mu, logvar, self.batch_size) # KL divergence
        

        return x_t_hat, mse, kld 

    
    def training_stage(self):

        self.kl_annealing.frange_cycle_linear(args.num_epoch)
        
        for epoch in range(self.args.num_epoch):
            
            
            train_loader = self.train_dataloader()
       
            n_iter = int ( len(train_loader.dataset) / args.batch_size )
            
            
            # teacher forcing
            adapt_TeacherForcing = True if random.random() < self.tfr else False
            
         
            epoch_loss = 0
            epoch_mse = 0
            epoch_kld = 0
            total  = 0
            
            # kl annealing
            beta = self.kl_annealing.get_beta()
            
            
            # training
            for idx, (img, label) in enumerate((pbar := tqdm(train_loader))):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                
                mse, kld = self.training_one_step(img, label, adapt_TeacherForcing, idx)
                
                # loss
                ave_mse = (mse / self.train_vi_len).detach().cpu().numpy()
                ave_kld = (kld / self.train_vi_len).detach().cpu().numpy() 
                ave_loss = ave_mse + ave_kld * beta
                epoch_loss += ave_loss
                epoch_mse += ave_mse
                epoch_kld += ave_kld

                # gradient descent
                loss = mse + kld * beta 
                loss.backward()
                self.optimizer_step() 

                # total += label.size(0) 
            

                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {:.2f}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {:.2f}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            # print(f"total: {n_iter}")
            
            # append training process list
            self.training_loss.append( epoch_loss / n_iter )
            self.training_mse_loss.append(epoch_mse / n_iter)
            self.training_kld_loss.append(epoch_kld / n_iter)
            self.training_tfr.append(self.tfr)
            self.training_kl_weight.append(beta)

  

            psnr_list = self.eval()
            # print(f"psnr list: {psnr_list}")
            ave_psnr =  sum(psnr_list) / len(psnr_list)
            # print(f"ave psnr: {ave_psnr}")
            ave_psnr = ave_psnr.detach().cpu()
      
            if ave_psnr > self.best_psnr or self.current_epoch % self.args.per_save==0:
                # save the last model
                self.best_psnr = ave_psnr
            
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))
                


            if self.current_epoch == self.args.num_epoch - 1 :
                torch.save({
                    'args': args,
                    "state_dict": self.state_dict(),
                    "optimizer": self.state_dict(),  
                    "lr"        : self.scheduler.get_last_lr()[0],
                    "tfr"       :   self.tfr,
                    "last_epoch": self.current_epoch,
                    "psnr" : self.best_psnr,
                    'last_epoch': self.current_epoch,
                    'ave_psnr': self.ave_psnr_list,
                    'train_loss': self.training_loss,
                    'train_mse_loss': self.training_mse_loss,
                    'train_kld_loss': self.training_kld_loss,
                    'train_kl_weight': self.training_kl_weight,
                    'train_kl_tfr': self.training_tfr},
                    '%s/training_process.pth' % args.save_root)
                                      

            self.ave_psnr_list.append(ave_psnr)


            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update(self.current_epoch)
            self.kl_annealing.update()
            
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        
        psnr_list = []
        for idx, (img, label) in enumerate((pbar := tqdm(val_loader, ncols=120))):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            psnr = self.val_one_step(img, label)

            self.tqdm_bar('val', pbar, psnr.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            
            psnr_list.append(psnr.detach().cpu())
        return psnr_list

    def training_one_step(self, img, label, adapt_TeacherForcing, idx):
        img = img.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        label = label.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
   
        mse_tol = 0
        kld_tol = 0
        for i in range(1, self.train_vi_len):
            p_t = label[i]

            if i == 1 or adapt_TeacherForcing:
                x_t_1 = img[i - 1]
            else:
                x_t_1  = x_t_hat
            x_t = img[i]
            

            # generate next frame
            x_t_hat, mse, kld = self.forward(p_t, x_t_1, x_t)

            # loss
            mse_tol += mse 
            kld_tol += kld 
        return mse_tol, kld_tol
        # TODO
        # raise NotImplementedError
    
    def val_one_step(self, img, label):
        img = img.permute(1, 0, 2, 3, 4) # change tensor into (B, seq, C, H, W)
        label = label.permute(1, 0, 2, 3, 4) # change tensor into (B, seq, C, H, W)

        pred_x = []
        pred_x.append(img[0])
 
        with torch.no_grad():
            for i in range(1, self.val_vi_len):

                if i == 1:
                    x_t_1 = img[i - 1]
                else:
                    x_t_1  = x_t_hat

                x_t = img[i]
                p_t = label[i]

                
                # generate next frame
                x_t_hat, _, _ = self.forward(p_t, x_t_1, x_t)
                
                # store pred
                pred_x.append(x_t_hat)
        pred_x = torch.stack(pred_x).permute(1, 0, 2, 3, 4)

        # generate git here to see the effect

        # calculate psnr
        pred_x = pred_x[0]
        # print(img.size())
        PSNR_LIST = []
        for i in range(1, self.val_vi_len):
            PSNR = Generate_PSNR(img[i][0], pred_x[i])
            PSNR_LIST.append(PSNR.detach().cpu())
        # print(f"PSNR_LIST: {PSNR_LIST}...")
        return sum(PSNR_LIST)/(len(PSNR_LIST)-1)

        # TODO
        # raise NotImplementedError
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self, epoch):
        if epoch >= self.tfr_sde: # start decay to not be forced
            decay_rate = self.tfr_d_step  # set the decay rate in args
            self.tfr = max(self.tfr - decay_rate, 0)  # ensure TFR does not go below 0
        # TODO
        #raise NotImplementedError
            
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr:.7f}" , refresh=False)
        if mode == 'val':
            pbar.set_postfix(psnr=float(loss), refresh=False)
        else:
            pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch,
            "psnr" : self.best_psnr 
        }, path)
        print(f"save best psnr {self.best_psnr} ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            print(f"loading checkpoint...")
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5], gamma=0.09)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch'] + 1

            self.best_psnr = checkpoint['psnr']
            # self.current_epoch = checkpoint['last_epoch']
            # self.ave_psnr_list = checkpoint['ave_psnr']
            # self.training_loss = checkpoint['train_loss']
            # self.training_mse_loss = checkpoint['train_mse_loss']
            # self.training_kld_loss = checkpoint['train_kld_loss']
            # self.training_kl_weight = checkpoint['train_kl_weight']
            # self.training_tfr = checkpoint['train_kl_tfr']


    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()



def main(args):
    
    os.makedirs(args.save_root, exist_ok=True)


    device_id = torch.cuda.current_device()
    print( f"device id: {device_id}, device name: {torch.cuda.get_device_name(device_id)}" )
    print(f'Using device: {torch.cuda.get_device_name(args.device)}')


    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=4)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")# BEST: 0.001
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=200,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=50,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=20,   help="The epoch that teacher forcing ratio start to decay") 
    parser.add_argument('--tfr_d_step',    type=float, default=0.05,  help="Decay step that teacher forcing ratio adopted") 
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")# BEST: 0.4
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=4,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=0.5,              help="")
    

    

    args = parser.parse_args()
    
    main(args)
