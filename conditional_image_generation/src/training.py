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
    """Computes cross-covariance loss between nuisance variables and
attribute predictions, so that latent code does note encode
attributes. Computes mean first.

    z: nuisance variables
    y: predicted labels
    bs: batch size

    """
    

    z = z.view(bs,-1)
    y = y.view(bs,-1)

    z = z - torch.mean(z, dim=0)
    y = y - torch.mean(y, dim=0)

    cov_matrix = torch.matmul(torch.t(z),y)

    cov_loss = torch.norm(cov_matrix.view(1,-1))/bs

    return cov_loss




class Trainer(object):

    def __init__(self, ae, ae_teacher, data, params):
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

        print '---- TEACHER INFO ---- '
        logger.info(ae_teacher)
        
        
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
            print '>>>>>>>>>> RELOADING TEACHER'
            reload_model(ae_teacher, params.ae_teacher_reload,
                         ['img_sz', 'img_fm', 'init_fm', 'n_layers', 'n_skip', 'attr', 'n_attr'])
            
            print 'reloaded teacher <<<<'  
                
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


    def autoencoder_step(self):
        """
        Train the autoencoder with cross-entropy loss.
        Train the encoder with discriminator loss.
        """
        data = self.data
        params = self.params



        self.ae.train()
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
        # with batchy
        if params.lambda_xcov >0:
            xcov_cost = compute_xcov(z_latent_1, y_pred, bs)
        else:        
            xcov_cost = Variable(torch.FloatTensor([0])) #


        xcov_loss =  xcov_cost * params.lambda_xcov # Variable(torch.FloatTensor([0])) #


            

        # autoencoder loss from reconstruction
        ae_cost = ((batch_x - dec_outputs[-1]) ** 2).mean() 
        loss = params.lambda_ae * ae_cost


        # print 'loss first', loss
        
        self.stats['rec_costs'].append(ae_cost.data[0])
        self.stats['attr_pred_costs'].append(attr_cost.data[0])
        self.stats['xcov_costs'].append(xcov_cost.data[0])


        # add first attribute prediction cost
        loss += attr_loss 

        
        # add cross-covariance loss
        if params.lambda_xcov > 0:
            loss += xcov_loss


        if params.freeze_encoder:
            # clip to 0 encoder updates(freeze)
            for p_ix, para in enumerate(self.ae.enc_layers.parameters()):
                para.grad[:] = 0

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
            ##      Encode/Decode with Teacher
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


            jacobian_cost = ((diff_ae - diff_ae_teacher.clone().detach())**2).mean()
            
            # print images

            # OUTDIR = '/home/jose/Documents/code/20180810_FN_jacobian/results_tmp/'
            # print_grid(dec_teacher_outputs[-1],   os.path.join(OUTDIR, 'dec_teacher_outputs.png'))
            # print_grid(dec_teacher_outputs_2[-1], os.path.join(OUTDIR, 'dec_teacher_outputs_2.png'))

            # print_grid(dec_outputs[-1], os.path.join(OUTDIR, 'dec_outputs.png'))
            # print_grid(dec_outputs_2[-1], os.path.join(OUTDIR, 'dec_outputs_2.png'))

            # print_grid(diff_ae, os.path.join(OUTDIR, 'diff_ae.png'))
            # print_grid(diff_ae_teacher, os.path.join(OUTDIR, 'diff_ae_teacher.png'))
            # raise

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
            flipped_labels_prediction_cost = Variable(torch.cuda.FloatTensor([0]))
            
            # latent code match cost
            latent_code_match_cost = Variable(torch.cuda.FloatTensor([0]))



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
        if params.lambda_jacobian >0:
            # self.ae_optimizer_enc.zero_grad()
            # self.ae_optimizer_dec.zero_grad()

            # add jacobian cost (difference of difference of output between student and teacher)
            loss += jacobian_loss

            loss += y_loss

            
        self.ae_optimizer_enc.zero_grad()
        self.ae_optimizer_dec.zero_grad()

        loss.backward() #retain_graph=True)

        # print 'loss second backward', loss
        if params.clip_grad_norm:
            clip_grad_norm(self.ae.parameters(), params.clip_grad_norm)
                
        self.ae_optimizer_enc.step()
        self.ae_optimizer_dec.step() 


        # add flipped labels prediction cost from reconstruction
        if params.lambda_flipped > 0 or params.lambda_latent_match > 0:
            print 'flipped...'
            self.ae_optimizer_enc.zero_grad()
            self.ae_optimizer_dec.zero_grad()

            
            loss = 0
            # print 'loss is 0 again', loss
            loss += flipped_labels_prediction_loss
            # print 'loss plus flipped', loss
            # add difference of latent codes
            loss += latent_code_match_loss
            # print 'loss plus latent', loss
            
            ########
            # Optimize steps 2 and 3
            # only affects decoder
            
            
            loss.backward()# retain_graph=True)
            # print 'loss second backward', loss
            if params.clip_grad_norm:
                clip_grad_norm(self.ae.parameters(), params.clip_grad_norm)

            if params.freeze_encoder_step2:
                # clip to 0 encoder updates(freeze)
                for p_ix, para in enumerate(self.ae.enc_layers.parameters()):
                    para.grad[:] = 0
            else:
                self.ae_optimizer_enc.step()

            self.ae_optimizer_dec.step() # step only for decoder


            
        

##########################            
    def step(self, n_iter):
        """
        End training iteration / print training statistics.
        """
        # average loss
        if len(self.stats['rec_costs']) >= 25:
            mean_loss = [
                ('Attr', 'attr_pred_costs', float(self.params.lambda_ttributes)),
                ('Rec', 'rec_costs', float(self.params.lambda_ae)),
                ('Xcov', 'xcov_costs', float(self.params.lambda_xcov)),
                ('jac', 'jacobian_costs', float(self.params.lambda_jacobian)),
                ('y', 'y_costs', float(self.params.lambda_y))
            ]
            logger.info(('%06i - ' % n_iter) +
                        '/ '.join(['%s : %2.3e (x%1.1e)' % (a, np.mean(self.stats[b]), c)
                                    for a, b,c in mean_loss if len(self.stats[b]) > 0]))
            del self.stats['rec_costs'][:]
            del self.stats['xcov_costs'][:]
            del self.stats['jacobian_costs'][:]
            del self.stats['y_costs'][:]

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
        if self.params.n_lat_dis:
            save(self.lat_dis, 'lat_dis')
        if self.params.n_ptc_dis:
            save(self.ptc_dis, 'ptc_dis')
        if self.params.n_clf_dis:
            save(self.clf_dis, 'clf_dis')

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


