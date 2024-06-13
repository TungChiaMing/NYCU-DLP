import matplotlib.pyplot as plt
import numpy as np

def frange_cycle(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
 
    period = n_epoch/n_cycle
     
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop and (int(i+c*period) < n_epoch):
            L[int(i+c*period)] = v
            # print(int(i+c*period))
            v += step
            i += 1
    return L    
def frange_linear(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
 
    period = n_epoch
    step = (stop - start) / (period * ratio)


    v , i = 0 , 0
    while v <= stop and (int(i) < n_epoch):
        L[int(i)] = v

        v += step
        i += 1
    return L    
    



# forget to save kl, just create the list
cyc = frange_cycle(0.0, 1.0, 200, 4, 0.5)
inc = frange_linear(0.0, 1.0, 200, 4, 0.5 )

 
class Result():
    def __init__(self):
        pass

    @staticmethod
    def plot_loss(epochs, loss, mse_loss, kld_loss, tfr, args):
     
     
    
        epochs = np.arange(0, len(loss))
        
        # print(kl_func)
        # Creating figure
        fig, ax1 = plt.subplots(figsize=(10,6))

        '''
        manually change here
        '''
        ax1.set_title(f'Without KL annealing Loss/Ratio Curve', fontsize=18) 


        color = 'tab:red'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(epochs, loss, color='tab:green', label='Loss')
        ax1.plot(epochs, mse_loss, color='tab:blue', label='MSE Loss')
        ax1.plot(epochs, kld_loss, color=color, label='KL Loss')
        
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='center right')
        ax1.set_ylim(0.0008, 0.01)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('Rate', color=color)  # we already handled the x-label with ax1
        ax2.plot(epochs, tfr, color='tab:green', linestyle='dotted', label='Teacher Forcing Ratio')

        '''
        manually change here
        '''
        # ax2.plot(epochs, inc, color='tab:blue', linestyle='dotted', label='KL weight')
        
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        
        plt.savefig('./{}/trainingCurve.png'.format(args.save_root))

    @staticmethod
    def plot_psnr(psnr, args):
        # Creating figure
        plt.subplots(figsize=(10,5))
        plt.title('Pre frame Quality (PSNR)', fontsize=18)

        plt.xlabel('Frame index', fontsize=14)
        plt.ylabel('PSNR', fontsize=14)
        # plt.yticks(range(0, 30 +1, 5))


        plt.plot(psnr, label=f'Avg_PSNR: {np.mean(psnr):.4f}')
    
        plt.legend(loc='upper right')
 
        plt.savefig('./{}/psnr.png'.format(args.save_root))