# Copyright 2018-present, Jose Lezama
#
# Modification of Fader Networks, Copyright (c) 2017-present, Facebook, Inc.
# 
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from logging import getLogger

from .utils import get_optimizer, clip_grad_norm, get_lambda, reload_model, reload_model2
from .model import get_attr_loss, flip_attributes

from print_images import print_grid

logger = getLogger()

def compute_xcov(z,y,bs):
    """computes cross-covariance loss between latent code and attributes
prediction, so that latent code does note encode attributes, compute
mean first."""
    # z: latent code
    # y: predicted labels
    # bs: batch size
    z = z.view(bs,-1)
    y = y.view(bs,-1)

    # print z.size(), y.size()

    # center matrices

    z = z - torch.mean(z, dim=0)
    y = y - torch.mean(y, dim=0)

    
    cov_matrix = torch.matmul(torch.t(z),y)

    cov_loss = torch.norm(cov_matrix.view(1,-1))/bs

    return cov_loss




class Evaluator(object):

    def __init__(self, ae,  ae_teacher, data, params):
        """
        Trainer initialization.
        """
        # data / parameters
        self.data = data
        self.params = params

        # modules
        self.ae = ae
        self.ae_teacher = ae_teacher

        # optimizers
        self.ae_optimizer_enc = get_optimizer(ae.enc_layers, params.ae_optimizer)
        self.ae_optimizer_dec = get_optimizer(ae.dec_layers, params.ae_optimizer)



        logger.info(ae)
        logger.info('%i parameters in the autoencoder. '
                    % sum([p.nelement() for p in ae.parameters()]))

        # reload pretrained models
        if params.ae_reload:
            print '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<', params.max_fm
            if int(params.max_fm) >=1064:
                print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
                print 'using reload_model2'
                
                reload_model2(ae, params.ae_reload,
                         ['img_sz', 'img_fm', 'init_fm', 'n_layers', 'n_skip', 'attr', 'n_attr'])
            else:
                reload_model(ae, params.ae_reload,
                         ['img_sz', 'img_fm', 'init_fm', 'n_layers', 'n_skip', 'attr', 'n_attr'])
            # reload teacher
            reload_model(ae_teacher, params.ae_teacher_reload,
                         ['img_sz', 'img_fm', 'init_fm', 'n_layers', 'n_skip', 'attr', 'n_attr'])

                
        # training statistics
        self.stats = {}
        self.stats['rec_costs'] = []
        self.stats['xcov_costs'] = []
        self.stats['attr_pred_costs'] = [] # to store attribute prediction
        self.stats['flipped_labels_prediction_costs'] = [] # to store prediction done on decoder output + encoder classifier
        self.stats['latent_code_match_costs'] = [] # to store difference between latent code and latent code of decoder output + encoder
        self.stats['jacobian_costs']  = [] # to store difference between difference of outputs for teacher and student
        self.stats['y_costs']  = [] # to store difference between teacher prediction and student prediction
        

        # best reconstruction loss / best accuracy
        self.best_loss = 1e12
        self.params.n_total_iter = 0

    def autoencoder_step(self, iterno=-1, epoch=-1):
        """
        Evaluate the autoencoder with cross-entropy loss.
        Evaluate the encoder with discriminator loss.
        """
        data = self.data
        params = self.params


        # evaluation mode
        self.ae.eval()


        bs = params.batch_size
        # batch / encode / decode
        batch_x, batch_y = data.train_batch(bs)

        #################
        ## Step 1: Normal encoder/decoder, with attribute label prediction
        enc_outputs, dec_outputs, y_pred = self.ae(batch_x, batch_y)

        z_latent_1 = enc_outputs[-1]


        # difference between y_pred and batch_y


        
        penalties = torch.clamp(1-y_pred*batch_y, min=0)



        attr_cost = torch.sum(penalties)/bs 
        attr_loss = attr_cost * params.lambda_ttributes

        # cross-covariance between z_latent_1 and y_pred

        if params.lambda_xcov >0:
            xcov_cost = compute_xcov(z_latent_1, y_pred, bs)
        else:        
            xcov_cost = Variable(torch.FloatTensor([0])) #


        xcov_loss =  xcov_cost * params.lambda_xcov # Variable(torch.FloatTensor([0])) #


            

        # autoencoder loss from reconstruction
        ae_cost = ((batch_x - dec_outputs[-1]) ** 2).mean() 
        loss = params.lambda_ae * ae_cost

        self.stats['rec_costs'].append(ae_cost.data[0])
        self.stats['attr_pred_costs'].append(attr_cost.data[0])
        self.stats['xcov_costs'].append(xcov_cost.data[0])


        # add first attribute prediction cost
        loss += attr_loss 

        # add cross-covariance loss
        if params.lambda_xcov > 0:
            loss += xcov_loss


        # compute costs for 2nd stage
        if  params.lambda_flipped > 0 or params.lambda_latent_match >0 or params.lambda_jacobian > 0:
            #################
            ## Step 2: Decode with flipped labels
            
            flipped = y_pred.clone()

            flip_idx = torch.randperm(bs).cuda()
            
            flipped = flipped[flip_idx,:]
            #flipped[:,23] = flipped[:,23]*-10

            
            dec_outputs_2 = self.ae.decode(enc_outputs, flipped) # returns an image created from same latent code but flipped labels
        else:

            # no need to compute all that
            # second prediction cost
            flipped_labels_prediction_cost = Variable(torch.FloatTensor([0]))
            
            # latent code match cost
            latent_code_match_cost = Variable(torch.FloatTensor([0]))

            jacobian_cost = Variable(torch.FloatTensor([0]))

        if params.lambda_jacobian >0:
            
            ################
            ##  Encode/Decode with Teacher
            self.ae_teacher.eval()
            for param in self.ae_teacher.parameters():
                param.requires_grad = False

                                
            
            enc_teacher_outputs, dec_teacher_outputs, y_teacher_pred = self.ae_teacher(batch_x, batch_y)

            teacher_flipped = y_teacher_pred.clone()

            teacher_flipped = teacher_flipped[flip_idx, :]
            #teacher_flipped[:,23] = teacher_flipped[:,23]*-10

            y_cost = ((y_pred - y_teacher_pred.detach()) **2).mean()
            
            dec_teacher_outputs_2 = self.ae_teacher.decode(enc_teacher_outputs, teacher_flipped)

            # difference between original output and flipped output should be the same for both decoders

            diff_ae = dec_outputs[-1] - dec_outputs_2[-1]
            diff_ae_teacher = dec_teacher_outputs[-1].detach() - dec_teacher_outputs_2[-1].detach()

            
            
            jacobian_cost = ((diff_ae - diff_ae_teacher)**2).mean()



            # print images

            if 0 and iterno==0 and ((epoch %10) ==0):
                DATAROOT = '/data/tmp'

                OUTDIR = '%s/%s' % (DATAROOT, params.outdir)
                os.system('mkdir -p %s' % OUTDIR)
                
                print_grid(dec_teacher_outputs[-1],   os.path.join(OUTDIR, '%04i_dec_teacher_outputs.png' % epoch))
                print_grid(dec_teacher_outputs_2[-1], os.path.join(OUTDIR, '%04i_dec_teacher_outputs_2.png' % epoch))
    
                print_grid(dec_outputs[-1], os.path.join(OUTDIR, '%04i_dec_outputs.png' % epoch))
                print_grid(dec_outputs_2[-1], os.path.join(OUTDIR, '%04i_dec_outputs_2.png' % epoch))
    
                print_grid(diff_ae, os.path.join(OUTDIR, '%04i_diff_ae.png' % epoch))
                print_grid(diff_ae_teacher, os.path.join(OUTDIR, '%04i_diff_ae_teacher.png' % epoch))

                print 'saved images to ', OUTDIR
            

        else:

            # no need to compute all that
            jacobian_cost = Variable(torch.FloatTensor([0]))
            y_cost = Variable(torch.FloatTensor([0]))
            

        if  params.lambda_flipped > 0 or params.lambda_latent_match >0:

            #################
            ## Step 3: Encode result of decoder with flipped labels
            enc_outputs_3 = self.ae.encode(dec_outputs_2[-1])
            
            # note that ae.encode is different from ae.forward, ae.forward returns latent code and attribute prediction (y_pred) separately whilst ae.encode does not sepparate
            
            z_all_3 = enc_outputs_3[-1]
            
            z_latent_3 = z_all_3[:,:-params.n_attr,:,:].contiguous()
            
            y_pred_3 = z_all_3[:,-params.n_attr:,:,:]

            y_pred_3 = torch.mean(y_pred_3.contiguous().view(bs, params.n_attr, -1), dim=2)
            
            
            flipped_labels = torch.sign(flipped)

            if 1:
                penalties_3 = torch.clamp(1-y_pred_3*flipped_labels, min=0)
                flipped_labels_prediction_cost =  torch.sum(penalties_3)/bs
            else:
                # Try L2 loss
                flipped_labels_prediction_cost = torch.sqrt(((y_pred_3 - flipped_labels) ** 2).sum())/bs


            
            # second prediction cost
            
            # latent code match cost
            latent_code_match_cost = torch.sqrt(((z_latent_1 - z_latent_3) ** 2).sum())/bs

        else:
            # no need to compute all that
            # second prediction cost
            flipped_labels_prediction_cost = Variable(torch.FloatTensor([0]))
            
            # latent code match cost
            latent_code_match_cost = Variable(torch.FloatTensor([0]))



        
        flipped_labels_prediction_loss = flipped_labels_prediction_cost * params.lambda_flipped
        latent_code_match_loss = latent_code_match_cost * params.lambda_latent_match

        jacobian_loss = jacobian_cost * params.lambda_jacobian
        y_loss = y_cost * params.lambda_y

        self.stats['flipped_labels_prediction_costs'].append(flipped_labels_prediction_cost.data[0])
        self.stats['latent_code_match_costs'].append(latent_code_match_cost.data[0])
        self.stats['jacobian_costs'].append(jacobian_cost.data[0])
        self.stats['y_costs'].append(y_cost.data[0])

        
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        #######
        ## Combine losses

        # add flipped labels prediction cost from reconstruction
        if params.lambda_flipped > 0 or params.lambda_latent_match > 0:
            loss = 0
            
            loss += flipped_labels_prediction_loss

            # add difference of latent codes
            loss += latent_code_match_loss

            # add jacobian cost (difference of difference of output between student and teacher)
            loss += jacobian_loss

            loss += y_loss

    def step(self, n_iter):
        """
        End training iteration / print training statistics.
        """
        # average loss
        if 1: # len(self.stats['rec_costs']) >= 25:
            mean_loss = [
                ('Attr', 'attr_pred_costs', float(self.params.lambda_ttributes)),
                ('Rec', 'rec_costs', float(self.params.lambda_ae)),
                ('Xcov', 'xcov_costs', float(self.params.lambda_xcov)),
                ('flip', 'flipped_labels_prediction_costs', float(self.params.lambda_flipped)),
                ('lat', 'latent_code_match_costs', float(self.params.lambda_latent_match)),
                ('jac', 'jacobian_costs', float(self.params.lambda_jacobian)),
                ('y', 'y_costs', float(self.params.lambda_y))
                 

            ]
            logger.info(('EVAL>> %06i - ' % n_iter) +
                        '/ '.join(['%s : %2.3e (x%1.1e)' % (a, np.mean(self.stats[b]), c)
                                    for a, b,c in mean_loss if len(self.stats[b]) > 0]))

        self.params.n_total_iter += 1

    def save_model(self, name):
        """
        Save the model.
        """
        def save(model, filename):
            path = os.path.join(self.params.dump_path, '%s_%s.pth' % (name, filename))
            logger.info('Saving %s to %s ...' % (filename, path))
            torch.save(model, path)
        save(self.ae, 'ae')

    def save_best_periodic(self, to_log):
        """
        Save the best models / periodically save the models.
        """
        if to_log['ae_loss'] < self.best_loss:
            self.best_loss = to_log['ae_loss']
            logger.info('Best reconstruction loss: %.5f' % self.best_loss)
            self.save_model('best_rec')
        if to_log['n_epoch'] % 5 == 0 and to_log['n_epoch'] > 0:
            self.save_model('periodic-%i' % to_log['n_epoch'])


