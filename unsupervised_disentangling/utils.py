import torch
import torch.nn as nn

from PIL import Image
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse


def to_img(x):
    x = 1-x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

def show(img, outfname=None):
    npimg = img.numpy()
    plt.imshow(npimg, interpolation='nearest', aspect='equal')
    if outfname is not None:
        Image.fromarray(npimg).convert('RGB').save(outfname)



# to show validation loss
def val_loss(tmp_model,  test_loader, hidden_size, train=False, teacher=None):
  tmp_model.eval()

  criterion = nn.MSELoss()


  if teacher:
      teacher.eval()

  count = 0
  loss_rec_all = 0 
  all_z = None

  loss_z_all = 0
  loss_xcov_all = 0
  loss_jacobian_all = 0

  
  for data in test_loader:
        img, y = data
        img = img.view(img.size(0), -1)
        img = img.cuda()
        y = y.cuda()
        # ===================forward=====================
        #output = model.decoder(Z)
        #output = model(img)
        z = tmp_model.encoder(img)
        tmp_output = tmp_model.decoder(z)  

        try:
          all_z = torch.cat((all_z,z))
          all_y = torch.cat((all_y,y))
        except:
          all_z = z.clone()
          all_y = y.clone()
        count+= 1
        
        loss_rec_all += criterion(tmp_output, img)
        
        if teacher:
            teacher_z = teacher.encoder(img)
            student_nuisance_z = z[:,:-hidden_size]
            student_factor_z = z[:,-hidden_size:]
      
            loss_z = torch.mean((teacher_z-student_factor_z)**2)
            loss_xcov = compute_xcov(student_nuisance_z, student_factor_z, teacher_z.size(0))

            # ================= start jacobian supervision ====
            # swap factor z
            swap_idx = torch.randperm(z.size(0)).cuda()
    
            student_z_swapped = torch.cat((student_nuisance_z,student_factor_z[swap_idx]),dim=1)
            teacher_z_swapped = teacher_z[swap_idx]
    
            
            swapped_tmp_output = tmp_model.decoder(student_z_swapped)
            swapped_teacher_output = teacher.decoder(teacher_z_swapped)
            teacher_output = teacher.decoder(teacher_z)
            
            diff_teacher = (teacher_output - swapped_teacher_output).clone().detach()
            diff_student = (tmp_output - swapped_tmp_output)
    
            if 0:#params.gauss_sigma >0:
                diff_teacher = diff_teacher.view([img.size(0),1,28,28])
                diff_student = diff_student.view([img.size(0),1,28,28])
    
                diff_student  = GBlur(diff_student)
                diff_teacher = GBlur_teacher(diff_teacher)
            
            jacobian_loss = torch.mean((diff_teacher-diff_student)**2)


            loss_z_all += loss_z
            loss_xcov_all += loss_xcov
            loss_jacobian_all += jacobian_loss
            
  if train:
    tmp_model.train()
    if teacher:
        teacher.train()

  loss_rec_all /= count
  loss_z_all /= count
  loss_xcov_all /= count
  loss_jacobian_all /= count

  if teacher:
      return loss_rec_all, loss_z_all, loss_jacobian_all, loss_xcov_all
  else:
      return loss_rec_all
 





""" #  Define Cross-covariance loss """
def compute_xcov(z,y,bs):
    """computes cross-covariance loss between latent code and attributes
prediction, so that latent code does note encode attributes, compute
mean first."""
    # z: latent code
    # y: predicted labels
    # bs: batch size
    z = z.contiguous().view(bs,-1)
    y = y.contiguous().view(bs,-1)

    # print z.size(), y.size()

    # center matrices

    z = z - torch.mean(z, dim=0)
    y = y - torch.mean(y, dim=0)

    
    cov_matrix = torch.matmul(torch.t(z),y)

    cov_loss = torch.norm(cov_matrix.view(1,-1))/bs

    return cov_loss




