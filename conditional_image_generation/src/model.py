# Copyright 2018-present, Jose Lezama
#
# Modification of Fader Networks, Copyright (c) 2017-present, Facebook, Inc.
# 
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


def build_layers(img_sz, img_fm, init_fm, max_fm, n_layers, n_attr, n_skip,
                 deconv_method, instance_norm, enc_dropout, dec_dropout):
    """
    Build auto-encoder layers.
    """
    assert init_fm <= max_fm
    assert n_skip <= n_layers - 1
    assert np.log2(img_sz).is_integer()
    assert n_layers <= int(np.log2(img_sz))
    assert type(instance_norm) is bool
    assert 0 <= enc_dropout < 1
    assert 0 <= dec_dropout < 1
    norm_fn = nn.InstanceNorm2d if instance_norm else nn.BatchNorm2d

    enc_layers = []
    dec_layers = []

    n_in = img_fm
    n_out = init_fm


    for i in range(n_layers):
        enc_layer = []
        dec_layer = []
        skip_connection = n_layers - (n_skip + 1) <= i < n_layers - 1
        if n_layers ==6:
            n_dec_in = n_out   + (n_attr if i<n_layers-1 else 0) # + (n_out if skip_connection else
        else:
            n_dec_in = n_out   + (n_attr if i<n_layers-2 else 0) # + (n_out if skip_connection else 0)

        n_dec_out = n_in  - (n_attr if (i==n_layers-1 and n_layers==7) else 0)

        # print n_layers
        # print 'n_in', n_in, 'n_out', n_out, 'n_dec_in', n_dec_in, 'n_dec_out', n_dec_out
        
        # encoder layer
        enc_layer.append(nn.Conv2d(n_in, n_out, 4, 2, 1))
        if i > 0:
            enc_layer.append(norm_fn(n_out, affine=True))
        enc_layer.append(nn.LeakyReLU(0.2, inplace=True))
        if enc_dropout > 0:
            enc_layer.append(nn.Dropout(enc_dropout))

        # decoder layer
        if deconv_method == 'upsampling':
            dec_layer.append(nn.UpsamplingNearest2d(scale_factor=2))
            dec_layer.append(nn.Conv2d(n_dec_in, n_dec_out, 3, 1, 1))
        elif deconv_method == 'convtranspose':
            dec_layer.append(nn.ConvTranspose2d(n_dec_in, n_dec_out, 4, 2, 1, bias=False))
        else:
            assert deconv_method == 'pixelshuffle'
            dec_layer.append(nn.Conv2d(n_dec_in, n_dec_out * 4, 3, 1, 1))
            dec_layer.append(nn.PixelShuffle(2))
        if i > 0:
            dec_layer.append(norm_fn(n_dec_out, affine=True))
            if dec_dropout > 0 and i >= n_layers - 3:
                dec_layer.append(nn.Dropout(dec_dropout))
            dec_layer.append(nn.ReLU(inplace=True))
        else:
            dec_layer.append(nn.Tanh())

        print max_fm, i, n_layers,  n_in, n_out
            
        # update
        n_in = n_out

        if max_fm >=1024:
            if i ==n_layers-3:
                print '----'
                n_out = 552
            else:
                if max_fm >= 1576 and i==n_layers-2:
                    print  'last_layer!'
                    n_out = max_fm
                else:
                    n_out = min(2 * n_out, max_fm)

        else:
            n_out = min(2 * n_out, max_fm)
        enc_layers.append(nn.Sequential(*enc_layer))
        dec_layers.insert(0, nn.Sequential(*dec_layer))
        
        
    # print dec_layers
    # raise
    return enc_layers, dec_layers


class AutoEncoder(nn.Module):

    def __init__(self, params):
        super(AutoEncoder, self).__init__()

        self.no_expand = params.no_expand
        
        self.img_sz = params.img_sz
        self.img_fm = params.img_fm
        self.instance_norm = params.instance_norm
        self.init_fm = params.init_fm
        self.max_fm = params.max_fm
        self.n_layers = params.n_layers
        self.n_skip = params.n_skip
        self.deconv_method = params.deconv_method
        self.dropout = params.dec_dropout
        self.attr = params.attr
        self.n_attr = params.n_attr
        self.ypred_type = params.ypred_type
        # print  self.n_attr, 'inside AE'
        # raise


        enc_layers, dec_layers = build_layers(self.img_sz, self.img_fm, self.init_fm,
                                              self.max_fm, self.n_layers, self.n_attr,
                                              self.n_skip, self.deconv_method,
                                              self.instance_norm, 0, self.dropout)
        self.enc_layers = nn.ModuleList(enc_layers)
        self.dec_layers = nn.ModuleList(dec_layers)

    def encode(self, x):
        assert x.size()[1:] == (self.img_fm, self.img_sz, self.img_sz)

        enc_outputs = [x]
        for layer in self.enc_layers:
            enc_outputs.append(layer(enc_outputs[-1]))

        assert len(enc_outputs) == self.n_layers + 1
        return enc_outputs

    def decode(self, enc_outputs, y):

        bs = enc_outputs[0].size(0)
        assert len(enc_outputs) == self.n_layers + 1

        
        assert y.size() == (bs, self.n_attr)



        dec_outputs = [enc_outputs[-1]]
        y = y.unsqueeze(2).unsqueeze(3)



        for i, layer in enumerate(self.dec_layers):
            size = dec_outputs[-1].size(2)
            # attributes

            if 0:# self.no_expand:

                y2 = Variable(torch.zeros((bs, self.n_attr,size,size))).cuda()
                y2[:,:,int(size/2),int(size/2)] = y
                
                input = [dec_outputs[-1], y2]

                
            else:
                
                input = [dec_outputs[-1], y.expand(bs, self.n_attr, size, size)]
            
                # print 'expand size', y.expand(bs, self.n_attr, size, size).size()
            # skip connection
            if 0 < i <= self.n_skip:
                input.append(enc_outputs[-1 - i])
            input = torch.cat(input, 1)

            dec_outputs.append(layer(input))

        # print '------------'
        assert len(dec_outputs) == self.n_layers + 1
        assert dec_outputs[-1].size() == (bs, self.img_fm, self.img_sz, self.img_sz)
        return dec_outputs

    def forward(self, x, y):
        enc_outputs = self.encode(x)
        bs = enc_outputs[0].size(0)
        
        # in this case, y will be predicted by the encoder
        z_all = enc_outputs[-1]
        # print z_all.size(), self.n_attr
        n_pred = self.n_attr
        y_pred = z_all[:,-n_pred:,:,:]

        z_latent = z_all[:,:-n_pred,:,:]

        enc_outputs[-1] = z_latent.contiguous()

        if self.ypred_type=='mean':
            y_pred = torch.mean(y_pred.contiguous().view(bs, self.n_attr, -1), dim=2)
        elif self.ypred_type == 'max':
            #print 'using max for y_pred!'
            y_pred, _ = torch.max(y_pred.contiguous().view(bs, self.n_attr, -1), dim=2)

        else:
            raise ValueError('Unknown ypred type')
            
            # print 'y_pred size', y_pred.size(), 'y size', y.size()

        #raise ValueError('termina')

        dec_outputs = self.decode(enc_outputs, y_pred)
        return enc_outputs, dec_outputs, y_pred


class LatentDiscriminator(nn.Module):

    def __init__(self, params):
        super(LatentDiscriminator, self).__init__()

        self.img_sz = params.img_sz
        self.img_fm = params.img_fm
        self.init_fm = params.init_fm
        self.max_fm = params.max_fm
        self.n_layers = params.n_layers
        self.n_skip = params.n_skip
        self.hid_dim = params.hid_dim
        self.dropout = params.lat_dis_dropout
        self.attr = params.attr
        self.n_attr = params.n_attr

        self.n_dis_layers = int(np.log2(self.img_sz))
        self.conv_in_sz = self.img_sz / (2 ** (self.n_layers - self.n_skip))
        self.conv_in_fm = min(self.init_fm * (2 ** (self.n_layers - self.n_skip - 1)), self.max_fm)
        self.conv_out_fm = min(self.init_fm * (2 ** (self.n_dis_layers - 1)), self.max_fm)

        # discriminator layers are identical to encoder, but convolve until size 1
        enc_layers, _ = build_layers(self.img_sz, self.img_fm, self.init_fm, self.max_fm,
                                     self.n_dis_layers, self.n_attr, 0, 'convtranspose',
                                     False, self.dropout, 0)

        self.conv_layers = nn.Sequential(*(enc_layers[self.n_layers - self.n_skip:]))
        self.proj_layers = nn.Sequential(
            nn.Linear(self.conv_out_fm, self.hid_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.hid_dim, self.n_attr)
        )

    def forward(self, x):

        conv_output = self.conv_layers(x)
        assert conv_output.size() == (x.size(0), self.conv_out_fm, 1, 1)
        return self.proj_layers(conv_output.view(x.size(0), self.conv_out_fm))




def get_attr_loss(output, attributes, flip, params):
    """
    Compute attributes loss.
    """
    assert type(flip) is bool
    k = 0
    loss = 0
    for (_, n_cat) in params.attr:
        # categorical
        x = output[:, k:k + n_cat].contiguous()
        y = attributes[:, k:k + n_cat].max(1)[1].view(-1)
        if flip:
            # generate different categories
            shift = torch.LongTensor(y.size()).random_(n_cat - 1) + 1
            y = (y + Variable(shift.cuda())) % n_cat
        loss += F.cross_entropy(x, y)
        k += n_cat
    return loss


def update_predictions(all_preds, preds, targets, params):
    """
    Update discriminator / classifier predictions.
    """
    assert len(all_preds) == len(params.attr)
    k = 0
    for j, (_, n_cat) in enumerate(params.attr):
        _preds = preds[:, k:k + n_cat].max(1)[1]
        _targets = targets[:, k:k + n_cat].max(1)[1]
        all_preds[j].extend((_preds == _targets).tolist())
        k += n_cat
    assert k == params.n_attr


def get_mappings(params):
    """
    Create a mapping between attributes and their associated IDs.
    """
    if not hasattr(params, 'mappings'):
        mappings = []
        k = 0
        for (_, n_cat) in params.attr:
            assert n_cat >= 2
            mappings.append((k, k + n_cat))
            k += n_cat
        assert k == params.n_attr
        params.mappings = mappings
    return params.mappings


def flip_attributes(attributes, params, attribute_id, new_value=None):
    """
    Randomly flip a set of attributes.
    """
    assert attributes.size(1) == params.n_attr
    mappings = get_mappings(params)
    attributes = attributes.data.clone().cpu()

    def flip_attribute(attribute_id, new_value=None):
        bs = attributes.size(0)
        i, j = mappings[attribute_id]
        attributes[:, i:j].zero_()
        if new_value is None:
            y = torch.LongTensor(bs).random_(j - i)
        else:
            assert new_value in range(j - i)
            y = torch.LongTensor(bs).fill_(new_value)
        attributes[:, i:j].scatter_(1, y.unsqueeze(1), 1)

    if attribute_id == 'all':
        assert new_value is None
        for attribute_id in range(len(params.attr)):
            flip_attribute(attribute_id)
    else:
        assert type(new_value) is int
        flip_attribute(attribute_id, new_value)

    return Variable(attributes.cuda())


