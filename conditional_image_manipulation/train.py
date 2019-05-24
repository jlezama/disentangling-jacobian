# Source code for training the face attribute manipulation model as
# featured in "Overcoming the Disentanglement vs Reconstruction
# Trade-off via Jacobian Supervision", J. Lezama, ICLR 2019.
#
# Copyright 2018-present, Jose Lezama
#
# Borrows heavily from Fader Networks, Copyright (c) 2017-present, Facebook, Inc.
# 
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import argparse
import torch

from src.loader import load_images, DataSampler
from src.utils import initialize_exp, bool_flag, attr_flag, check_attr
from src.model import AutoEncoder
from src.training import Trainer
from src.evaluation2 import Evaluator2

# parse parameters
parser = argparse.ArgumentParser(description='Images autoencoder')
parser.add_argument("--name", type=str, default="default",
                    help="Experiment name")
parser.add_argument("--img_sz", type=int, default=256,
                    help="Image sizes (images have to be squared)")
parser.add_argument("--img_fm", type=int, default=3,
                    help="Number of feature maps (1 for grayscale, 3 for RGB)")
parser.add_argument("--attr", type=attr_flag, default="Smiling,Male",
                    help="Attributes to classify")
parser.add_argument("--instance_norm", type=bool_flag, default=False,
                    help="Use instance normalization instead of batch normalization")
parser.add_argument("--init_fm", type=int, default=32,
                    help="Number of initial filters in the encoder")
parser.add_argument("--max_fm", type=int, default=512,
                    help="Number maximum of filters in the autoencoder")
parser.add_argument("--no_expand", type=int, default=0,
                    help="do not expand prediction")
parser.add_argument("--n_layers", type=int, default=6,
                    help="Number of layers in the encoder / decoder")
parser.add_argument("--n_skip", type=int, default=0,
                    help="Number of skip connections")
parser.add_argument("--deconv_method", type=str, default="convtranspose",
                    help="Deconvolution method")
parser.add_argument("--outdir", type=str, default="model",
                    help="output dir")
parser.add_argument("--hid_dim", type=int, default=512,
                    help="Last hidden layer dimension for discriminator / classifier")
parser.add_argument("--dec_dropout", type=float, default=0.,
                    help="Dropout in the decoder")
parser.add_argument("--lambda_ae", type=float, default=1,
                    help="Autoencoder loss coefficient")
parser.add_argument("--lambda_ttributes", type=float, default=1,
                    help="Attributes prediction loss coefficient")
parser.add_argument("--lambda_flipped", type=float, default=1.0,
                    help="flipped prediction loss coefficient")
parser.add_argument("--lambda_latent_match", type=float, default=10,
                    help="latent code match loss coefficient")
parser.add_argument("--lambda_lat_dis", type=float, default=0.0001,
                    help="Latent discriminator loss feedback coefficient")
parser.add_argument("--lambda_xcov", type=float, default=0.1, help="xcov loss coefficient")
parser.add_argument("--lambda_jacobian", type=float, default=1.0, help="jacobian loss coefficient")
parser.add_argument("--lambda_y", type=float, default=100.0, help="same prediction cost coefficient")
parser.add_argument("--v_flip", type=bool_flag, default=False,
                    help="Random vertical flip for data augmentation")
parser.add_argument("--h_flip", type=bool_flag, default=True,
                    help="Random horizontal flip for data augmentation")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size")
parser.add_argument("--ae_optimizer", type=str, default="adam,lr=0.0002",
                    help="Autoencoder optimizer (SGD / RMSprop / Adam, etc.)")
parser.add_argument("--dis_optimizer", type=str, default="adam,lr=0.0002",
                    help="Discriminator optimizer (SGD / RMSprop / Adam, etc.)")
parser.add_argument("--clip_grad_norm", type=float, default=5,
                    help="Clip gradient norms (0 to disable)")
parser.add_argument("--n_epochs", type=int, default=100,
                    help="Total number of epochs")
parser.add_argument("--epoch_size", type=int, default=50000,
                    help="Number of samples per epoch")
parser.add_argument("--ae_reload", type=str, default="",
                    help="Reload a pretrained encoder")
parser.add_argument("--ae_teacher_reload", type=str, default=None,
                    help="Reload a pretrained jacobian teacher encoder")
parser.add_argument("--debug", type=bool_flag, default=False,
                    help="Debug mode (only load a subset of the whole dataset)")
parser.add_argument("--freeze_encoder", type=bool_flag, default=False,
                    help="Wether to freeze encoder weights always.")
parser.add_argument("--freeze_encoder_step2", type=bool_flag, default=False,
                    help="Wether to freeze encoder weights on step 2.")
params = parser.parse_args()

# check parameters
check_attr(params)
assert len(params.name.strip()) > 0
assert params.n_skip <= params.n_layers - 1
assert params.deconv_method in ['convtranspose', 'upsampling', 'pixelshuffle']
assert not params.ae_reload or os.path.isfile(params.ae_reload)

# add number of attributes to size of autoencoder
params.max_fm = params.max_fm + params.n_attr
params.max_fm_orig  = params.max_fm

if not params.ae_teacher_reload:
    params.ae_teacher_reload = params.ae_reload

    

# initialize experiment / load dataset
DATAROOT = '/data/tmp'
logger = initialize_exp(params)
data, attributes = load_images(params)


train_data = DataSampler(data[0], attributes[0], params)
valid_data = DataSampler(data[1], attributes[1], params)

# build the Student model
ae = AutoEncoder(params).cuda()


# build the Teacher model
params2 = params  
params2.max_fm = 512 + params.n_attr
ae_teacher = AutoEncoder(params2).cuda()

params.max_fm  = params.max_fm_orig

# trainer / evaluator
trainer = Trainer(ae, ae_teacher, train_data, params)

evaluator = Evaluator2(ae, ae_teacher, valid_data, params)



for n_epoch in range(params.n_epochs):

    logger.info('Starting epoch %i...' % n_epoch)

    for eee in range(25):
        evaluator.autoencoder_step(iterno=eee, epoch=n_epoch)
    evaluator.step(n_epoch)

    
    for n_iter in range(0, params.epoch_size, params.batch_size):

        # autoencoder training
        trainer.autoencoder_step()

        # print training statistics
        trainer.step(n_iter)

    
    
    
    if (n_epoch % 10) == 0:
        resultsdir = '%s/%s' % (DATAROOT, params.outdir)
        os.system('mkdir -p %s' % resultsdir)
        trainer.save_model('%s/epoch_%04i.pth' % (resultsdir, n_epoch))
    logger.info('End of epoch %i.\n' % n_epoch)
