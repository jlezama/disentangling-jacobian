""" Code for ICLR 2019 submission 
"Overcoming the Disentanglement vs Reconstruction Trade-off via Jacobian Supervision"

Copyright 2018-present Jose Lezama

This script is for reproducing the unsupervised experiments of Section 3 on MNIST.
"""

import os
import numpy as np
import time
from quantitative_evaluation import compute_quant


# Set hyperparameters
num_epochs = 100
lambda_z = 0.25 # \lambda_y in paper
lambda_jacobian =  0.1 # \lambda_{diff} in paper

lambda_xcov = 1e-3

use_bn = 0
    
max_h_dim = 17


# Adam parameters
wd = 1e-6
lr = 3e-4


# TRAIN FIRST TEACHER
start_time = time.time()
outdirs = []

results_dir = 'results_progressive'

outdir = 'z_%1.1e_jac_%1.1e_xcov_%1.1e_bn_%i/' % ( lambda_z, lambda_jacobian, lambda_xcov, use_bn)
os.system('mkdir -p %s' % os.path.join(results_dir, outdir))

argstr = "--outdir %s --num_epochs %i --wd %f --lr %f"  % (outdir, num_epochs, wd, lr)

command = "python train_teacher.py %s > %s" % (argstr, os.path.join(results_dir, outdir, 'log_teacher.txt'))


teacher_fname = os.path.join(results_dir, outdir, 'teacher.pth')
    


print command
os.system(command)
print 'elapsed time:', (time.time()-start_time)/60.0
print '--------'


# TRAIN FURTHER STUDENTS
for student_h_dim in range(3,max_h_dim):
    start_time = time.time()
    outdir = 'z_%1.1e_jac_%1.1e_xcov_%1.1e_bn_%i/h_dim_%i' % (lambda_z, lambda_jacobian, lambda_xcov, use_bn, student_h_dim)
    os.system('mkdir -p %s' % os.path.join(results_dir, outdir))

    argstr = "--teacher %s --student_h_dim %i --lambda_z %f --lambda_jacobian %f --lambda_xcov %f --outdir %s --num_epochs %i --wd %f --lr %f --use_student_batchnorm %i"  % (teacher_fname, student_h_dim, lambda_z, lambda_jacobian, lambda_xcov, outdir, num_epochs, wd, lr, use_bn)
    command = "python train_student.py %s  > %s" % (argstr,  os.path.join(results_dir, outdir, 'log.txt'))
    print command
    os.system(command)
    print 'elapsed time:', (time.time()-start_time)/60.0
    print '--------'

    # update teacher
    teacher_fname = os.path.join(results_dir, outdir, 'student.pth')
                
    
                
# compute quantitative measures
for student_h_dim in range(2,max_h_dim):
    if student_h_dim ==2:
        outdir = 'z_%1.1e_jac_%1.1e_xcov_%1.1e_bn_%i/' % ( lambda_z, lambda_jacobian, lambda_xcov, use_bn)
    else:
        outdir = 'z_%1.1e_jac_%1.1e_xcov_%1.1e_bn_%i/h_dim_%i' % ( lambda_z, lambda_jacobian, lambda_xcov, use_bn, student_h_dim)
    rec, flip = compute_quant(os.path.join(results_dir, outdir), student_h_dim, use_bn)
        
