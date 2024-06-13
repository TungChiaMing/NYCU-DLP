import matplotlib.pyplot as plt
import imageio
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid, save_image
import numpy as np

def tqdm_bar(mode, pbar, loss, lr, current_epoch):
    pbar.set_description(f"({mode}) Epoch {current_epoch}, lr:{lr:.7f}" , refresh=False)
    if mode == 'test':
        pbar.set_postfix(psnr=float(loss), refresh=False)
    else:
        pbar.set_postfix(loss=float(loss), refresh=False)
    pbar.refresh()


class Result():
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def plot_result(test_acc, new_test_acc, fig_path):
         
        plt.figure(figsize=(10, 5))

        plt.title('Accuracy curve')

        plt.plot(test_acc, label='test')
        plt.plot(new_test_acc, label='new_test')
        
      
        # plt.yticks(np.arange(0.1, 0.8 + 0.2, 0.2))
        plt.legend(loc='lower right')
        
        #max_point = np.argmax(test_acc)
        #plt.annotate(f'{np.max(test_acc):.2f}', (max_point, test_acc[max_point]), textcoords="offset points", xytext=(-10,10), ha='center', color='blue')

        #max_point = np.argmax(new_test_acc)
        #plt.annotate(f'{np.max(new_test_acc):.2f}', (max_point, new_test_acc[max_point]), textcoords="offset points", xytext=(-10,10), ha='center', color='orange')

        plt.xlabel('Epoch', fontsize=14)   
        plt.ylabel('Accuracy', fontsize=14)
        plt.savefig(fig_path)
