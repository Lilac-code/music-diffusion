
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import numpy as np

# the unet implementation from https://github.com/lucidrains/denoising-diffusion-pytorch
from denoising_diffusion_pytorch import Unet


# class that on the fly creates a batch from the dataset
class Pianoroll(Dataset):
    def __init__(self, rolls, ratio):
        super(Pianoroll).__init__()
        self.rolls = rolls
        self.num_steps = 100
        self.ratio = ratio
    def __len__(self):
        return len(self.rolls) * self.num_steps

    def __getitem__(self, index):
        roll = self.rolls[index//self.num_steps]
        t = index%self.num_steps
        beta = (t+1)/self.num_steps
        noisy = np.random.binomial(1, roll*(1-beta)+self.ratio*beta)
        return noisy.astype(np.float32), roll.astype(np.float32), t

# create dataloader_train that feeds the network 50 epochs of the training data with shuffling
# segments is the list of piano-roll segment to train the network on
# ratio is the ratio of 1s to the size of the piano-roll segment (#rows by #columns)
pr = Pianoroll(rolls=segments, ratio=0.032)
batch_size = 200
dataloader_train = DataLoader(pr, batch_size=batch_size, shuffle=True)

unet = Unet(dim=48, channels=1, resnet_block_groups=3, dim_mults=(1, 2, 4, 4))
unet.to('cuda')

params = list(unet.parameters())
optimizer = Adam(params, lr=5e-5)

epochs = 50

loss_function = nn.L1Loss()

optimizer.zero_grad()

for epoch in range(epochs):
    for step, batch in enumerate(dataloader_train):
        batch_roll = batch[0].cuda()
        batch_time = batch[2].cuda()
        
        predicted_x0 = unet(batch_roll, batch_time)
        
        loss = loss_function(predicted_x0.float(), batch[1].float().cuda())
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()


# sampling method
total_num_steps = 100
beta = 1.0
ratio = 0.032

tmp = np.zeros((384, 56)) # size of a piano-roll segment
noisy_intial = np.random.binomial(1, tmp*(1-beta)+ratio*beta)
noisy = np.copy(noisy_intial)

for i in range(total_num_steps):
    predicted_x0 = unet(torch.from_numpy(noisy.astype(np.float32)).cuda('cuda'), 
                        torch.tensor(total_num_steps-i-1, dtype=torch.float32).cuda('cuda')).cpu().detach().numpy()

    threshold = 0.5
    predicted_x0 = predicted_x0 >= threshold

    beta = (total_num_steps-i)/total_num_steps

    delta = predicted_x0 ^ noisy_intial
    mask = np.random.binomial(1, delta*beta)
    noisy = predicted_x0*(1-mask) + noisy_intial * mask

