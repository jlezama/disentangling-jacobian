# -*- coding: utf-8 -*-
"""Jacobian MNIST Autoencoder

"""

import sys, os
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from models import autoencoder, autoencoder_nobatchnorm
from utils import show, to_img, val_loss
    
from PIL import Image
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse

# parse parameters
parser = argparse.ArgumentParser(description='Images autoencoder')
parser.add_argument("--outdir", type=str, default="test",
                    help="output dir")
parser.add_argument("--num_epochs", type=int, default=100,
                    help="Total number of epochs")
parser.add_argument("--wd", type=float, default=1e-5, help="weight decay")
parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
params = parser.parse_args()


params.outdir = os.path.join('results_progressive', params.outdir)


""" Download MNIST Dataset
"""

train_dataset = dsets.MNIST(root='./data',
                           train=True,
                           transform=transforms.ToTensor(),
                           download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())



  

""" --------- Train the Autoencoder ------------ """
input_size    = 784   # The image size = 28 x 28 = 784
hidden_size   = 2   # The number of nodes at the hidden layer
num_classes   = 10    # The number of output classes. In this case, from 0 to 9
num_epochs    = params.num_epochs     # The number of times entire dataset is trained
batch_size    = 128   # The size of input data took for one iteration
learning_rate = params.lr  # The speed of convergence

show_every = max(1,int(num_epochs/10))
# initialize model
model = autoencoder(h_dim=hidden_size).cuda()



# Data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# initialize optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters()  , lr=learning_rate, weight_decay=params.wd)


teacher_fname = os.path.join(params.outdir, 'teacher.pth')
if not os.path.isfile(teacher_fname):
  for epoch in range(num_epochs):
      train_loss = 0
      count = 0
      for data in train_loader:
          img, y = data
          img = img.view(img.size(0), -1)
          img = img.cuda()
          y = y.cuda()
          # ===================forward=====================
          #output = model.decoder(Z)
          #output = model(img)
          z = model.encoder(img)
          output = model.decoder(z)
          loss = criterion(output, img)
  
          # ===================backward====================
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          train_loss += loss.data.item()
          count += 1
        
      # ===================log========================
      train_loss /= count
      if epoch % show_every == 0:

          val = val_loss(model, test_loader, hidden_size, train=True)
          
          print('epoch [{}/{}], loss:{:.4f}, val:{:.4f}, train_loss:{:.4f}'
            .format(epoch + 1, num_epochs, loss.data.item(), val.data.item(), train_loss))
          pic = to_img(output.cpu().data)
          show(pic[0][0] )
 
  torch.save(model.state_dict(), teacher_fname)

else:
   # load teacher model
   checkpoint = torch.load(teacher_fname)
   model.load_state_dict(checkpoint)
   



"""
------- Sample 2D latent code -------
"""

model.eval()
N = 10 # number of images per size
range_ = 2 # range of exploration

largepic = torch.zeros((N*28,N*28))

i=0
for z1 in np.linspace(-range_,range_,N):
  j=0
  for z2 in np.linspace(-range_,range_,N):
      z = torch.Tensor(np.asarray([z1, z2])).cuda()
      
      output = model.decoder(z).view(1,1,28,28)
      pic = to_img(output.cpu().data)
      
      #print(i,i+28,j,j+28)    
      largepic[i:i+28,j:j+28] = pic
      j+=28
  i+=28
      
    
outfname = os.path.join(params.outdir, 'teacher_output.png')
show(255-largepic*255, outfname )



