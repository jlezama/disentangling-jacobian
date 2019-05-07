# -*- coding: utf-8 -*-

import sys

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

import os
import argparse

from models import autoencoder, autoencoder_nobatchnorm
from utils import show, to_img, val_loss


import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt

import numpy as np


from mnist import mnist


mnist_model = mnist(pretrained=True).cuda()



"""# Download MNIST Dataset

MNIST is a huge database of handwritten digits (i.e. 0 to 9) that is often used in image classification.
"""

train_dataset = dsets.MNIST(root='./data',
                           train=True,
                           transform=transforms.ToTensor(),
                           download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

torch.manual_seed(2)

batch_size    = 128   # The size of input data took for one iteration


# Data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                           shuffle=True, drop_last=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False, drop_last=True)

# initialize optimizer
criterion = nn.MSELoss()




def compute_quant(outdir, student_h_dim, use_bn):

    input_size    = 784   # The image size = 28 x 28 = 784
    hidden_size   = student_h_dim    # The number of nodes at the hidden layer
    num_classes   = 10    # The number of output classes. In this case, from 0 to 9
    
    
    ### loading different pytorch version
    import torch._utils
    try:
        torch._utils._rebuild_tensor_v2
    except AttributeError:
        def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
            tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
            tensor.requires_grad = requires_grad
            tensor._backward_hooks = backward_hooks
            return tensor
        torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
    
    
    """# Create student model"""
    
    student_h_dim = student_h_dim
    
    
    if student_h_dim ==2:
        fname = os.path.join(outdir, 'teacher.pth')
        student_model = autoencoder(h_dim=student_h_dim).cuda()
    
    else:
        fname = os.path.join(outdir, 'student.pth' )
        if use_bn:
            student_model = autoencoder(h_dim=student_h_dim).cuda()
        else:
            student_model = autoencoder_nobatchnorm(h_dim=student_h_dim).cuda()
    
    print('loading student model', fname)
    
    checkpoint = torch.load(fname)
    student_model.load_state_dict(checkpoint, strict=False)
                 
    
    
    
    #=========================
    # generate last_z
    student_model.eval()


    # student_model.train()
    # test_loader = train_loader

    accs_train = 0.0
    count = 0.0
    all_z = None
    all_y = None
    for data in train_loader:
            img, y = data
            img = img.view(img.size(0), -1)
            img = Variable(img).cuda()
            
    
            y = Variable(y).cuda()
            
            last_z = student_model.encoder(img)
    
            out = student_model.decoder(last_z)
    
            scores =  mnist_model(out)
            y_pred = scores.max(1)[1]
    
            acc = ((y_pred==y).sum().float()/float(batch_size)).item()
    
            
            accs_train += acc
            count +=1.0
    
    
            
            if all_z is None:
                all_z = last_z.data
                all_y = y.data
            else:
                all_z = torch.cat((all_z, last_z))
                all_y = torch.cat((all_y, y))
                #=========================
        
    #znp =  all_z.data.cpu()[:,-2:].numpy()
    ix = all_z.size(1)-2
    
    print 'ix is ', ix
    
    print 'acc train is', accs_train / count
    
    
    znp =  all_z.data.cpu()[:,ix:ix+2].numpy()
    ynp = all_y.data.cpu().numpy()
    
    

    ###########################################################################
    
    # run test
    all_test_z = None
    all_test_y = None
    
    all_flip_acc = 0.0
    count = 0.0
    count2 = 0.0
    all_ae_loss = 0.0

    student_model.train()

    
    for iter_, data in enumerate(test_loader):
            img, y = data
            img = img.view(img.size(0), -1)
            img = Variable(img).cuda()
            
    
            y = Variable(y).cuda()
            
            last_z = student_model.encoder(img)
    
            out = student_model.decoder(last_z)

            # print out.max(), out.min(), img.max(), img.min()

            
            ae_loss = ((out-img)**2).mean().item()
            all_ae_loss +=ae_loss
            if all_test_z is None:
                all_test_z = last_z.data
                all_test_y = y.data
            else:
                all_test_z = torch.cat((all_test_z, last_z))
                all_test_y = torch.cat((all_test_y, y))
                #=========================
 

            # run on generated image
            scores =  mnist_model(out)
            y_pred = scores.max(1)[1]
            y = y_pred
    
            # swap labels
            flip_idx = torch.randperm(batch_size).cuda()
    
            flipped_y = y.clone()[flip_idx]
    
            are_different = (flipped_y !=y).nonzero()
    
    
            
            flipped_z = last_z.clone()
    
            flipped_z[:,ix:ix+2] = flipped_z[flip_idx,ix:ix+2]
    
            flipped_out = student_model.decoder(flipped_z)
    
            flipped_scores =  mnist_model(flipped_out)
            flipped_y_pred = flipped_scores.max(1)[1]
    
            flip_acc = ((flipped_y_pred[are_different]==flipped_y[are_different]))
    
    
            all_flip_acc += float(flip_acc.sum().item())
            count += float(flip_acc.size(0))
            count2 +=1


            if iter_<0:

                
                out_pic = out.clone().detach().view(-1,28).cpu()
                flipped_out_pic = flipped_out.clone().detach().view(-1,28).cpu()
                flipped_target_pic = out[flip_idx,:].clone().detach().view(-1,28).cpu()
                outfname = os.path.join(outdir, 'eval_out_%i.png' % iter_)
                show2(out_pic*255, outfname )
                outfname = os.path.join(outdir, 'eval_out_flipped_%i.png' % iter_)
                show2(flipped_out_pic*255, outfname )
                outfname = os.path.join(outdir, 'eval_flip_target_out_%i.png' % iter_)
                show2(flipped_target_pic*255, outfname )
                print 'saved out', outfname
                print y[0:20]
                print flipped_y_pred[0:20]
                print y[flip_idx][0:20]
                print flip_idx[0:20]
                print (flipped_y_pred==y[flip_idx]).sum(), batch_size, flip_acc.sum()



            
    print 'FLIP TEST', all_flip_acc/count
    # print 'REC TEST', all_ae_loss/count2

    rec = all_ae_loss/count2
    flip = all_flip_acc/count
    
    result_txt =  'QUANTITATIVE RESULTS >> %s latent code dimension: %i reconstruction:  %f swaps ok: %f' % (outdir, student_h_dim, rec, flip)

    print result_txt

    f = open(os.path.join(outdir, 'log_quant.txt'), 'w')
    f.write(result_txt + '\n')
    f.close()
    
    znp_test =  all_test_z.data.cpu()[:,ix:ix+2].numpy()
    ynp_test = all_test_y.data.cpu().numpy()
    
    # test_score = regr.score(znp_test, ynp_test)
    # print 'test score', test_score

    return float(rec), float(flip)

                
    
