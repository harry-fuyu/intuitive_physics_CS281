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
from scipy.stats.stats import pearsonr 

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

pyro.enable_validation(True)
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)
# Enable smoke test - run the notebook cells on CI.
smoke_test = 'CI' in os.environ
    

class TrajectoryDataset(Dataset):
    def __init__(self, status):
        self.copy = []
        trans = transforms.ToTensor()
        if status == "train":
            path = PATH_DATA + "train/"
            number = 2000
        else:
            path = PATH_DATA + "test/"
            number = 500

        for j in range(number):
            image_name = path + "example_" + str(j) + ".png"
            img = Image.open(image_name)
            img = img.resize((28, 28), Image.ANTIALIAS)
            img = img.convert('1')
            img = torch.from_numpy(np.array(img)).float()
            self.copy.append(img)
        
        self.len = len(self.copy)
        
    def __getitem__(self, index):
        image = self.copy[index]
        return image

    def __len__(self):
        return self.len


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        # setup the two linear transformations used
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(self.latent_dim,1024)
        self.fc21 = nn.Linear(1024, 784)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        z = self.softplus(self.fc1(z))
        bern_img = self.sigmoid(self.fc21(z))
        return bern_img


class Encoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.rnn = nn.RNN(2 * 28, self.hidden_dim)
        self.fc21 = nn.Linear(NUM_SLICES * self.hidden_dim, self.latent_dim)
        self.fc22 = nn.Linear(NUM_SLICES * self.hidden_dim, self.latent_dim)
        self.softplus = nn.Softplus()

    def forward(self, x):
        self.hidden = torch.zeros((1, BATCH_SIZE, self.hidden_dim))
        x = x.view(NUM_SLICES,BATCH_SIZE,-1).float()
        x, self.hidden = self.rnn(x, self.hidden)
        x = x.view((BATCH_SIZE,NUM_SLICES * self.hidden_dim)) 
        z_loc = self.fc21(x)
        z_scale = torch.exp(self.fc22(x))
        return z_loc, z_scale


class VAE(nn.Module):
    def __init__(self, latent_dim, hidden_dim, use_cuda=False):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.encoder = Encoder(self.hidden_dim, self.latent_dim)
        self.decoder = Decoder(self.latent_dim)

    def model(self, x):
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.latent_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.latent_dim)))
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            bern_img = self.decoder.forward(z)
            pyro.sample("obs", dist.Bernoulli(bern_img).to_event(1), obs=x.reshape(-1, 784))

    def guide(self, x):
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            z_loc, z_scale = self.encoder.forward(x)
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    def reconstruct_img(self, x):
        z_loc, z_scale = self.encoder(x)
        z = dist.Normal(z_loc, z_scale).sample()
        loc_img = self.decoder(z)
        return loc_img


def train(svi, train_loader):
    epoch_loss = 0.
    for x in train_loader:
        epoch_loss += svi.step(x)
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train


def evaluate(svi, test_loader):
    test_loss = 0.
    for x in test_loader:
        test_loss += svi.evaluate_loss(x)
    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test


def run(svi, train_loader, test_loader):
    train_elbo = []
    test_elbo = []
    # training loop
    for epoch in range(NUM_EPOCHS):
        total_epoch_loss_train = train(svi, train_loader)
        train_elbo.append(-total_epoch_loss_train)
        print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

        if epoch % TEST_FREQUENCY == 0:
            total_epoch_loss_test = evaluate(svi, test_loader)
            test_elbo.append(-total_epoch_loss_test)
            print("[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test))

    print("Done Training!")
    plt.plot(train_elbo)
    plt.xlabel("Epoch")
    plt.ylabel("train_ELBO")
    plt.savefig(PATH_OUTPUT + "train_elbo_epoch.png" )
    plt.close()

    plt.plot(test_elbo)
    plt.xlabel("Epoch")
    plt.ylabel("test_ELBO")
    plt.savefig(PATH_OUTPUT + "train_elbo_epoch.png" )
    plt.close()


def show_reconstruction(vae, TestLoader, iter):
    counter = 0
    for image in TestLoader:
        recon_img = vae.reconstruct_img(image)
        image = image.reshape(BATCH_SIZE, 28, 28)
        
        for i in range(BATCH_SIZE):
            ori_img = np.asarray(image[i].detach())

            fig = plt.figure()
            plt.imshow(ori_img)
            plt.title("original_trajectory")
            
            image_file_name = PATH_OUTPUT + "original" + str(counter) + "_" + str(i) + ".png"
            plt.savefig(image_file_name, bbox_inches='tight', pad_inches=0)
            img = Image.open(image_file_name).convert('L')
            img.save(image_file_name, format='PNG')
            plt.close()
        
        for i in range(BATCH_SIZE):
            img = np.asarray(recon_img[i].detach())

            fig = plt.figure()
            plt.title("reconstructed_trajectory")
            plt.imshow(img.reshape(28,28))
            image_file_name = PATH_OUTPUT + "recon" + str(counter) + "_" + str(i) + ".png"
            plt.savefig(image_file_name, bbox_inches='tight', pad_inches=0)
            img = Image.open(image_file_name).convert('L')
            img.save(image_file_name, format='PNG')
            plt.close()

        counter += 1
        if counter == 20:
            break


PATH = "/Users/chopinboy/Desktop/pyro/sliced_trajectory/"
PATH_DATA = PATH + "images/"
PATH_OUTPUT = PATH + "VAE/"
PATH_COMPARE = PATH + "compare/"

NUM_IMAGES = 2000
NUM_SLICES = 14
BATCH_SIZE = 4
LR = 0.001
NUM_EPOCHS = 180
TEST_FREQUENCY = 5

input_dim = 28 * 28
hidden_dim = 100
latent_dim = 2

train_data = TrajectoryDataset("train")
test_data = TrajectoryDataset("test")
TrainLoader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
TestLoader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


#############################################
# Initiate and train

# pyro.clear_param_store()
# vae = VAE(latent_dim, hidden_dim)

# adam_args = {"lr": LR}
# optimizer = Adam(adam_args)

# svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

# run(svi, TrainLoader, TestLoader)

# show_reconstruction(vae, TestLoader, 20)

# torch.save(vae.state_dict(), PATH_OUTPUT + "VAE_02.pt")



#############################################
# extrapolate latent space

# vae2 = VAE(latent_dim, hidden_dim)
# vae2.load_state_dict(torch.load(PATH_OUTPUT + "VAE_02.pt"))

# nx = 20
# ny = 20
# x_values = np.linspace(-2, 2, nx)
# y_values = np.linspace(-2, 2, ny)

# canvas = np.empty((28 * nx, 28 * ny))
# for i, xi in enumerate(x_values):
#     for j, yi in enumerate(y_values):
#         latent_tensor = np.zeros((BATCH_SIZE, latent_dim))
#         for k in range(BATCH_SIZE):
#                 latent_tensor[k] = np.asarray((xi,yi))
#             # print(model2(latent_tensor))
#         latent_tensor = torch.from_numpy(latent_tensor).float()
#         reconstructed = np.asarray(vae2.decoder(latent_tensor).detach())[0]
#         canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = reconstructed.reshape(28, 28)


# plt.figure(figsize=(10, 10))        
# Xi, Yi = np.meshgrid(x_values, y_values)
# plt.imshow(canvas, origin="upper",cmap = "gray")
# plt.tight_layout()
# plt.savefig(PATH_OUTPUT + "latent_VAE.png")
# plt.close()


#############################################
# test correspondence with true parameters


def standardize(x):
    return (x - np.mean(x)) / np.sqrt(np.var(x))


def compare_image(TestLoader, iter, speed, angle):
    vae3 = VAE(latent_dim, hidden_dim)
    vae3.load_state_dict(torch.load(PATH_OUTPUT + "VAE_02.pt"))
    counter = 0
    for image in TestLoader:
        recon_img = vae3.reconstruct_img(image)
        t = torch.zeros(4,2)
        t[:,0] = torch.from_numpy(speed[counter * BATCH_SIZE:((counter + 1) * BATCH_SIZE)])
        t[:,1] = torch.from_numpy(angle[counter * BATCH_SIZE:((counter + 1) * BATCH_SIZE)])
        truth_img = vae3.decoder(t)
        print(truth_img)

        for i in range(BATCH_SIZE):
            ori_img = np.asarray(truth_img[i].view(28,28).detach())
            fig = plt.figure()
            plt.imshow(ori_img)
            plt.title("true_para_reconstruction")
            
            image_file_name = PATH_COMPARE + "true_para" + str(counter) + "_" + str(i) + ".png"
            plt.savefig(image_file_name, bbox_inches='tight', pad_inches=0)
            img = Image.open(image_file_name).convert('L')
            img.save(image_file_name, format='PNG')
            plt.close()
        
        for i in range(BATCH_SIZE):
            img = np.asarray(recon_img[i].view(28,28).detach())

            fig = plt.figure()
            plt.title("reconstructed")
            plt.imshow(img.reshape(28,28))
            image_file_name = PATH_COMPARE + "recon" + str(counter) + "_" + str(i) + ".png"
            plt.savefig(image_file_name, bbox_inches='tight', pad_inches=0)
            img = Image.open(image_file_name).convert('L')
            img.save(image_file_name, format='PNG')
            plt.close()

        if counter == iter:
            break
        
        counter += 1


test = np.arange(1,100,1)


def get_correlation(TestLoader, iter, speed, angle):
    master1 = torch.zeros(iter)
    master2 = torch.zeros(iter)
    vae3 = VAE(latent_dim, hidden_dim)
    vae3.load_state_dict(torch.load(PATH_OUTPUT + "VAE_02.pt"))

    counter = 0
    for image in TestLoader:
        z = vae3.encoder(image)

        master1[counter * BATCH_SIZE : (counter + 1) * BATCH_SIZE] = z[0][:,0]
        # master2[counter * BATCH_SIZE : (counter + 1) * BATCH_SIZE] = z[1][:,0]
        if counter == iter:
            break
    
    print(pearsonr(np.asarray(master1.detach()), angle[0:iter]))
    print(pearsonr(np.asarray(master2.detach()), speed[0:iter]))



truth = pd.DataFrame.from_csv("/Users/chopinboy/Desktop/pyro/sliced_trajectory/test_initial_parameters.csv")
truth.columns = ["y","vx","vy"]
vx, vy = np.asarray(truth["vx"]), np.asarray(truth["vy"])
speed = np.sqrt(np.square(vx) + np.square(vy))
angle = np.arctan(np.divide(vy,vx))
speed = standardize(speed)
angle = standardize(angle)

compare_image(TestLoader, 200, speed, angle)
# get_correlation(TestLoader, 200, speed, angle)
