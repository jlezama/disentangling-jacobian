# -*- coding: utf-8 -*-
"""Jacobian MNIST Autoencoder
"""

import sys
sys.version


"""# Import PyTorch"""

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np

import os
import argparse

from models import autoencoder, autoencoder_nobatchnorm
from utils import show, to_img, val_loss, compute_xcov


# parse parameters
parser = argparse.ArgumentParser(description='Images autoencoder')
parser.add_argument("--student_h_dim", type=int, default=5,
                    help="latent code size for student")
parser.add_argument("--outdir", type=str, default="test",
                    help="output dir")
parser.add_argument("--teacher", type=str, default="test",
                    help="teacher location")
parser.add_argument("--lambda_z", type=float, default=1.0, help="lambda z")
parser.add_argument("--lambda_jacobian", type=float, default=1.0, help="lambda jacobian")
parser.add_argument("--lambda_xcov", type=float, default=0.0001, help="lambda xcov")

parser.add_argument("--num_epochs", type=int, default=100,
                    help="Total number of epochs")
parser.add_argument("--wd", type=float, default=1e-5, help="weight decay")
parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
parser.add_argument("--use_student_batchnorm", type=int, default=0,
                    help="Wether to use batchnorm in students")
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

  

"""Train the Autoencoder."""

input_size    = 784   # The image size = 28 x 28 = 784
hidden_size   = params.student_h_dim -1   # The number of nodes at the hidden layer
num_classes   = 10    # The number of output classes. In this case, from 0 to 9
num_epochs    = params.num_epochs     # The number of times entire dataset is trained
batch_size    = 128   # The size of input data took for one iteration
learning_rate = params.lr  # The speed of convergence

show_every = max(1,int(num_epochs/10))


# Data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# initialize teacher model
if params.student_h_dim > 3 and (not params.use_student_batchnorm):
    teacher_model = autoencoder_nobatchnorm(h_dim=hidden_size).cuda()
else:
    teacher_model = autoencoder(h_dim=hidden_size).cuda() # first teacher always uses batchnorm




# initialize optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    teacher_model.parameters()  , lr=learning_rate, weight_decay=params.wd)


# load teacher model
checkpoint = torch.load(params.teacher)
teacher_model.load_state_dict(checkpoint)
   

"""# Create student model"""
student_h_dim = params.student_h_dim


if params.use_student_batchnorm:
    student_model = autoencoder(h_dim=student_h_dim).cuda()
else:
    student_model = autoencoder_nobatchnorm(h_dim=student_h_dim).cuda()

"""# Copy parameters from teacher to student"""

for k in student_model.state_dict().keys():
        if student_model.state_dict()[k].size() != teacher_model.state_dict()[k].size():
            # copy manually                                                                                                                                                              
            their_model = teacher_model.state_dict()[k]
            my_model = student_model.state_dict()[k]
            sz = their_model.size()
            print(k, my_model.size(), their_model.size())

            if k.startswith('encoder'):
              my_model[-sz[0]:] = their_model[:]
            else:
              #print(my_model[:,-sz[1]:].size(),their_model.size())
              my_model[:,-sz[1]:] = their_model[:]

 
        else:
            # copy straight                                                                                                                                                              
            student_model.state_dict()[k].copy_(teacher_model.state_dict()[k])

"""# Try student model"""

for data in train_loader:
        img, y = data
        img = img.view(img.size(0), -1)
        img = img.cuda()
        y = y.cuda()
        # ===================forward=====================
        #output = model.decoder(Z)
        #output = model(img)
        z = student_model.encoder(img)
        output = student_model.decoder(z)  

        z2 = teacher_model.encoder(img)
        
        #print(z[0],z2[0])
        
        pic = to_img(output.cpu().data)
        show(pic[0][0] )
        #print( y[0])
        break



"""# Finetune student with Jacobian"""

# initialize optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    student_model.parameters(), lr=params.lr, weight_decay=params.wd)

lambda_z = params.lambda_z
lambda_jacobian = params.lambda_jacobian
lambda_xcov = params.lambda_xcov

teacher_model.train()
student_model.train()

student_fname = os.path.join(params.outdir, 'student.pth')
last_z_fname = os.path.join(params.outdir, 'last_z.pth')


if not os.path.isfile(student_fname):
  for epoch in np.arange(num_epochs):
    for data in train_loader:
        img, y = data
        img = img.view(img.size(0), -1)
        img = img.cuda()
        

        y = y.cuda()
        
        # ===================forward=====================
        #output = model.decoder(Z)
        #output = model(img)
        z = student_model.encoder(img)
        student_output = student_model.decoder(z)
        loss = 0

        loss_rec = criterion(student_output, img)

        loss += loss_rec
        
        teacher_z = teacher_model.encoder(img)
    
        student_nuisance_z = z[:,:-hidden_size]
        student_factor_z = z[:,-hidden_size:]
      
        loss_z = torch.mean((teacher_z-student_factor_z)**2)
      
        loss += loss_z * lambda_z# cost for same factor z prediction

        # ======== xcov loss ========
        loss_xcov = compute_xcov(student_nuisance_z, student_factor_z, teacher_z.size(0))
    
        loss += loss_xcov * lambda_xcov

        
        # ================= start jacobian supervision ====
        # swap factor z
        swap_idx = torch.randperm(z.size(0)).cuda()
        
        student_z_swapped = torch.cat((student_nuisance_z,student_factor_z[swap_idx]),dim=1)
        teacher_z_swapped = teacher_z[swap_idx]

        
        swapped_student_output = student_model.decoder(student_z_swapped)
        swapped_teacher_output = teacher_model.decoder(teacher_z_swapped)
        teacher_output = teacher_model.decoder(teacher_z)
        
        
        diff_teacher = (teacher_output - swapped_teacher_output).clone().detach()
        diff_student = (student_output - swapped_student_output)

        jacobian_loss = torch.mean((diff_teacher-diff_student)**2)
         
        loss += jacobian_loss * lambda_jacobian
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #break
        
    # ===================log========================
    if epoch % show_every == 0:

        print('epoch [{}/{}], loss:{:.4f}, loss_rec:{:.4f} loss_z:{:.4f}({:1.1e}), jacobian_loss:{:.4f}({:1.1e}), xcov_loss:{:.4f}({:1.1e})'
          .format(epoch + 1, num_epochs, loss.data.item(), loss_rec.data.item(), loss_z.data.item(), lambda_z, jacobian_loss.data.item(), lambda_jacobian, loss_xcov.data.item(), lambda_xcov))

        loss_rec_val, loss_z_val,  loss_jacobian_val, loss_xcov_val = val_loss(student_model, test_loader, hidden_size, train=True, teacher=teacher_model)

        
        print('EVAL >>> epoch [{}/{}],  loss_rec:{:.4f} loss_z:{:.4f}({:1.1e}), jacobian_loss:{:.4f}({:1.1e}), xcov_loss:{:.4f}({:1.1e})'
          .format(epoch + 1, num_epochs,  loss_rec_val.data.item(), loss_z_val.data.item(), lambda_z, loss_jacobian_val.data.item(), lambda_jacobian, loss_xcov_val.data.item(), lambda_xcov))

        pic = to_img(student_output.cpu().data)
        show(pic[0][0] )

  torch.save(student_model.state_dict(), student_fname)
  last_z = z
  torch.save(last_z, last_z_fname)
else:
    print('loading student jacobian model', student_fname)
    checkpoint = torch.load(student_fname)
    student_model.load_state_dict(checkpoint)
    last_z = torch.load(last_z_fname)


#=========================

"""# Show exploration grid for student (with jacobian)"""
student_model.eval()

### Explore nuisance variables
N = 10 # number of images per size
range_1_min = last_z.data[:,0].min() # range of exploration
range_1_max = last_z.data[:,0].max()
range_2_min = last_z.data[:,1].min()
range_2_max = last_z.data[:,1].max()


largepic = torch.zeros((28,N*28))

for ix in range(5):
  j=0
  for z1 in np.linspace(range_1_min, range_1_max, N):

    z = last_z[ix].clone() # get latent part from training example
        
    z[0] = z1
        
    output = student_model.decoder(z).view(1,1,28,28)
    pic = to_img(output.cpu().data)



    largepic[0:28,j:j+28] = pic[0][0]
    j += 28
    
                                                      
    
  outfname = os.path.join(params.outdir, 'student_traverse_nuisance_%i.png' % ix)
  show(255-largepic*255, outfname )


# explore factors
### Explore nuisance variables
N = 10 # number of images per size
range_1_min = -2 # last_z.data[:,-2].min() # range of exploration
range_1_max = 2 # last_z.data[:,-2].max()
range_2_min = -2 # last_z.data[:,-1].min()
range_2_max = 2 # last_z.data[:,-1].max()


largepic = torch.zeros((N*28,N*28))

for ix in range(5):
  i=0
  for z1 in np.linspace(range_1_min, range_1_max, N):
    j=0
    for z2 in np.linspace(range_2_min, range_2_max, N):
        z = last_z[ix].clone() # get latent part from training example
        
        z[-2] = z1
        z[-1] = z2

        # z[0] = z1
        # z[1] = z2

        output = student_model.decoder(z).view(1,1,28,28)
        pic = to_img(output.cpu().data)

        largepic[i:i+28,j:j+28] = pic[0][0]
        j+=28
    i+=28
                                                      
    
  outfname = os.path.join(params.outdir, 'student_traverse_factors_%i.png' % ix)
  show(255-largepic*255, outfname )


