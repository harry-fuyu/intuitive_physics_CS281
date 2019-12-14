import numpy as np
import torch
import torchvision
import torchvision.datasets as dset
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.externals import joblib 
import pandas as pd

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

pyro.enable_validation(True)
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)
# Enable smoke test - run the notebook cells on CI.
smoke_test = 'CI' in os.environ

PATH_DATA = '/Users/chopinboy/Desktop/pyro/partial_trajectories/'
NUM_IMAGES = 200
NUM_PARTIAL_TRAJECTORY = 8
    

class TrajectoryDataset(Dataset):
    def __init__(self):
        self.copy = []
        self.guide = []
        trans = transforms.ToTensor()
        guide_csv = np.asarray(pd.DataFrame.from_csv(PATH_DATA + "initial_parameters.csv"))
        guide_df = guide_csv[:,1:3]

        for j in range(NUM_IMAGES):
            folder_name = PATH_DATA + "example_" + str(j)
            image = torch.zeros((NUM_PARTIAL_TRAJECTORY,28,28))
            for i in range(NUM_PARTIAL_TRAJECTORY):
                img = Image.open(folder_name + "/partial_" + str(i) + ".png")
                img = img = img.convert('1')
                image[i] = torch.from_numpy(np.array(img))
                label = np.zeros((2,1))
                label[0] = guide_df[i][0]
                label[1] = guide_df[i][1]
                label = torch.from_numpy(label)
            self.copy.append((image,label))
        
        self.len = len(self.copy)
        
    def __getitem__(self, index):
        image, guide = self.copy[index]
        return image, guide

    def __len__(self):
        return self.len


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.RNN(self.input_dim, self.hidden_dim)
        self.decoder = nn.RNN(self.hidden_dim,self.input_dim)
        self.latent = None
        
    def forward(self,x):
        self.latent = None
        self.hidden = torch.zeros((1, BATCH_SIZE, hidden_dim))
        x = x.view(NUM_PARTIAL_TRAJECTORY,BATCH_SIZE,input_dim)
        x, self.hidden = self.encoder(x, self.hidden)
        x = x.view((BATCH_SIZE,NUM_PARTIAL_TRAJECTORY,self.hidden_dim))
        self.latent = x

        self.new_hidden = torch.zeros((1, NUM_PARTIAL_TRAJECTORY, self.input_dim))
        x, self.new_hidden = self.decoder(x, self.new_hidden)
        x = x.view((BATCH_SIZE,NUM_PARTIAL_TRAJECTORY,-1))
        x = x[:,-1,:].view(BATCH_SIZE,28,28)
        return x


def calculate_loss(model, data, pair):
    loss = 0
    recon_img = model(data)
    standard = data[:,-1,:,:]
    latent = model.latent

    for i in range(BATCH_SIZE):
        loss += torch.dist(recon_img[i],standard[i],2)
        
        for j in range(NUM_PARTIAL_TRAJECTORY):
            latent_block = latent[i][j]
            loss += torch.dist(latent_block, pair.float())

    return loss


def train(model, train_loader,lr):
    loss_history = []
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.
        for x, pair in train_loader:
            loss = calculate_loss(model, x, pair)
            epoch_loss += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if epoch % 10 == 0:
            print("Number of Epochs = ", epoch, ", Loss = ", epoch_loss.item())
        loss_history.append(epoch_loss)
    
    print("Done Training!")
    plt.plot(loss_history)
    plt.xlabel("Num_epoch")
    plt.ylabel("loss")
    plt.savefig( "/Users/chopinboy/Desktop/pyro/guided_recon/loss_against_epoch.png" )
    plt.close()


NUM_EPOCHS = 200
LR = 0.01
BATCH_SIZE = 4

Trajectories = TrajectoryDataset()
TrajectoryLoader = torch.utils.data.DataLoader(Trajectories, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

input_dim = 28 * 28
hidden_dim = 2

model = AutoEncoder(input_dim, hidden_dim)
optimizer = optim.SGD(model.parameters(), lr=LR)

train(model, TrajectoryLoader, LR)


dataiter = iter(TrajectoryLoader)
images = dataiter.__next__()


counter = 0
for image, label in TrajectoryLoader:
    recon_img = model(image)
    image = image[:,-1,:,:].reshape(BATCH_SIZE, 28, 28)
    
    for i in range(BATCH_SIZE):
        ori_img = np.asarray(image[i].detach())

        fig = plt.figure()
        plt.imshow(ori_img)
        plt.title("original_trajectory")
        
        image_file_name = "/Users/chopinboy/Desktop/pyro/guided_recon/" + "original" + str(counter) + "_" + str(i) + ".png"
        plt.savefig(image_file_name, bbox_inches='tight', pad_inches=0)
        img = Image.open(image_file_name).convert('L')
        img.save(image_file_name, format='PNG')
        plt.close()
    
    for i in range(BATCH_SIZE):
        img = np.asarray(recon_img[i].detach())

        fig = plt.figure()
        plt.title("reconstructed_trajectory")
        plt.imshow(img)
        image_file_name = "/Users/chopinboy/Desktop/pyro/guided_recon/" + "recon" + str(counter) + "_" + str(i) + ".png"
        plt.savefig(image_file_name, bbox_inches='tight', pad_inches=0)
        img = Image.open(image_file_name).convert('L')
        img.save(image_file_name, format='PNG')
        plt.close()

    counter += 1
    if counter == 10:
        break

torch.save(model.state_dict(), "/Users/chopinboy/Desktop/pyro/guided_model.pt")



#############################################
# extrapolate latent space

# model2 = AutoEncoder(input_dim, hidden_dim)
# model2.load_state_dict(torch.load("/Users/chopinboy/Desktop/pyro/model.pt"))


# counter = 0
# for image in TrajectoryLoader:
#     recon_img = model2(image)
#     image = image[:,-1,:,:].reshape(BATCH_SIZE, 28, 28)
    
#     for i in range(BATCH_SIZE):
#         img = np.asarray(recon_img[i].detach())

#         fig = plt.figure()
#         plt.title("reconstructed_trajectory")
#         plt.imshow(img)
#         plt.show()
#         plt.close()
