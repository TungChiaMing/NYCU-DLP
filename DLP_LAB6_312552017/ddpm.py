from diffusers import DDPMScheduler
import torch
from utils import tqdm_bar
from tqdm import tqdm

# Set the noise scheduler
noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def forward(clean_images, text_label):
        '''
        add noise
        '''
        # x0 ~ q(x0)
        # sample clean image
        # handle by dataloader

        # t ~ Uniform(0, num_train_timesteps)
        # sample a random timestep for each image
        bs = clean_images.shape[0]
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,)).long().to(device)


        # ε ~ N(0, 1)
        # sample noise to add to the images
        noise = torch.randn_like(clean_images)


        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        return noisy_images, timesteps, noise

from evaluator import evaluation_model
TA_evaluation_model = evaluation_model()

def train_step(train_dataloader, model, args, train_loss, epoch):
    model.train()
    tol_loss = 0

    for idx, (clean_images, text_label) in enumerate((pbar := tqdm(train_dataloader))):
        clean_images = clean_images.to(device) 
        text_label = text_label.to(device)

        # noise
        noisy_images, timesteps, noise = forward(clean_images, text_label)


        # predict noise
        noise_pred = model(noisy_images, timesteps, text_label).sample
        
        # loss
        loss = args.loss_fn(noise_pred, noise)
        tol_loss += loss.item()
        
        # gradient descent
        args.optimizer.zero_grad()
        loss.backward(loss)
        args.optimizer.step()
        args.scheduler.step()
        tqdm_bar('train', pbar, loss.detach().cpu(), args.scheduler.get_last_lr()[0], epoch)
        
    train_loss.append(tol_loss / len(train_dataloader.dataset))
   
    return train_loss 

        
from torchvision.utils import make_grid, save_image
def val_step(test_dataloader, model, args, acc_list, epoch, fname):
    test_labels_list = []

    for idx, label in enumerate(test_dataloader):
        test_labels_list.append(label)

    test_labels_list = torch.cat(test_labels_list, dim=0).to(device)

    num_samples = len(test_labels_list)

    image = torch.randn(num_samples, 3, 64, 64).to(device)
    for i, timestep in enumerate(noise_scheduler.timesteps):
        with torch.no_grad():
            noise_residual = model(image, timestep, test_labels_list).sample

        image = noise_scheduler.step(noise_residual, timestep, image).prev_sample

    generated_img = (image / 2 + 0.5).clamp(0, 1)
    save_image(make_grid(generated_img, nrow=8), f"{args.save_root}/{fname}_{epoch}.png")
    
    
    # acc
    acc = TA_evaluation_model.eval(image, test_labels_list)
    acc_list.append(acc)

    return acc_list
