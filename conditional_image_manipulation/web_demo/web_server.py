#!/usr/bin/env python
"""

Very simple HTTP server for demoing facial attribute manipulation, as
shown in the paper "Overcoming the Disentanglement vs Reconstruction
Trade-off via Jacobian Supervision", J. Lezama, ICLR 2019.

Based on interpolation code from Fader Networks https://github.com/facebookresearch/FaderNetworks.

Usage::
    ./web-server.py [<port>]

"""
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import SocketServer


# for interpolation
import os
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib.image


import sys
sys.path.append('../')
from src.logger import create_logger
from src.loader import load_images, DataSampler
from src.utils import bool_flag

from os import curdir, sep

from aux import *

class S(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        if self.path == "/":
                
            self._set_headers()

            params.offset = 1106
            
            interpolations, y_pred = run_interpolations(params, test_data)
            outfname = compute_grid(interpolations, params)

            html = create_html_result(outfname, y_pred, params)

            self.wfile.write(html)

        elif self.path.startswith('/get_image'):

            self._set_headers()

            print self.path
            params.offset = int(self.path.split('fname=')[1])

            
            interpolations, y_pred = run_interpolations(params, test_data)
            outfname = compute_grid(interpolations, params)

            html = create_html_result(outfname, y_pred, params)

            self.wfile.write(html)            
            
        sendReply = False
	if self.path.endswith(".png"):
	    mimetype='image/png'
	    sendReply = True


	if sendReply == True:
	    #Open the static file requested and send it
	    f = open(curdir + sep + self.path) 
	    self.send_response(200)
	    self.send_header('Content-type',mimetype)
	    self.end_headers()
	    self.wfile.write(f.read())
	    f.close()
            
                
    def do_HEAD(self):
        self._set_headers()


    def do_POST(self):
        # Doesn't do anything with posted data
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = self.rfile.read(content_length) # <--- Gets the data itself
        self._set_headers()

        # parse post data
        alphas = np.zeros(40)
        for i in range(40):
            tmp = post_data.split('attr_%i=' % i)[1].split('&')[0]
            alphas[i] = float(tmp)
        

        interpolations, y_pred = run_interpolations(params, test_data, alphas=alphas)
        outfname = compute_grid(interpolations, params)

        html = create_html_result(outfname, alphas, params)
        
        
        self.wfile.write(html)

        
def run(server_class=HTTPServer, handler_class=S, port=80):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print 'Starting httpd...'
    httpd.serve_forever()


####################################################################
# INTERPOLATION

# parse parameters
parser = argparse.ArgumentParser(description='Attributes swapping')
parser.add_argument("--model_path", type=str, default="",
                    help="Trained model path")
parser.add_argument("--outdir", type=str, default="",
                    help="out dir suffix")
parser.add_argument("--dataset", type=str, default="test",
                    help="dataset type: train, val, test")
parser.add_argument("--port", type=str, default="9999",
                    help="http server port")
parser.add_argument("--mode", type=str, default="grid",
                    help="alpha mode, mult or grid")
parser.add_argument("--n_images", type=int, default=1,
                    help="Number of images to modify")
parser.add_argument("--offset", type=int, default=6,
                    help="First image index")
parser.add_argument("--n_interpolations", type=int, default=10,
                    help="Number of interpolations per image")
parser.add_argument("--alpha_mult", type=float, default=100,
                    help="How much multiply alpha by") 
parser.add_argument("--alpha_min", type=float, default=1,
                    help="Min interpolation value")
parser.add_argument("--alpha_max", type=float, default=1,
                    help="Max interpolation value")
parser.add_argument("--plot_size", type=int, default=5,
                    help="Size of images in the grid")
parser.add_argument("--selected_attr", type=str, default="0",
                    help="selected attribute")
parser.add_argument("--row_wise", type=bool_flag, default=True,
                    help="Represent image interpolations horizontally")
parser.add_argument("--output_path", type=str, default="output.png",
                    help="Output path")
params = parser.parse_args()

# check parameters
assert os.path.isfile(params.model_path), params.model_path
assert params.n_images >= 1 and params.n_interpolations >= 2

# patch to load model trained with newer pytorch version
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
 
logger = create_logger(None)
ae = torch.load(params.model_path).eval()

# restore main parameters
params.debug = False
params.batch_size = 32
params.v_flip = False
params.h_flip = False
params.img_sz = ae.img_sz
params.attr = ae.attr
params.n_attr = ae.n_attr


# load dataset
data, attributes = load_images(params)
#test_data = DataSampler(data[2], attributes[2], params)

if params.dataset == 'train':
    data_ix = 0
elif params.dataset == 'val':
    data_ix = 1
elif params.dataset == 'test':
    data_ix = 2
    
test_data = DataSampler(data[data_ix], attributes[data_ix], params)


def get_interpolations(ae, images, attributes, params, alphas):
    """
    Reconstruct images / create interpolations
    """
    ae.eval()
    
    assert len(images) == len(attributes)
    enc_outputs = ae.encode(images)

    # separate latent code and attribute prediction
    bs = enc_outputs[0].size(0)

    z_all = enc_outputs[-1] # full latent code

    n_pred = params.n_attr
    

    y_pred = z_all[:,-n_pred:,:,:]

    z_latent = z_all[:,:-n_pred,:,:]

    enc_outputs[-1] = z_latent.contiguous()


    y_pred = torch.mean(y_pred.contiguous().view(bs, params.n_attr, -1), dim=2)


    outputs = []

    # original image / reconstructed image / interpolations
    new_image = ae.decode(enc_outputs, y_pred)[-1]

    outputs.append(images)
    outputs.append(new_image)


    y_pred_tmp = y_pred.clone()

    if alphas is not None:
        print 'fixing alphas:', alphas
        for attr in range(40):
           y_pred_tmp[:,attr] =  alphas[attr]

    outputs.append(ae.decode(enc_outputs, y_pred_tmp)[-1])

    
    # return stacked images
    return torch.cat([x.unsqueeze(1) for x in outputs], 1).data.cpu(), y_pred.data.cpu().numpy().tolist()[0]


def run_interpolations(params, test_data, alphas=None):
    interpolations = []
    
    for k in range(0, params.n_images, 100):
        i = params.offset + k
        j = params.offset + min(params.n_images, k + 100)
        images, attributes = test_data.eval_batch(i, j)
        generated_images, y_pred = get_interpolations(ae, images, attributes, params, alphas)
        interpolations.append(generated_images)
    
    interpolations = torch.cat(interpolations, 0)

    return interpolations, y_pred

def get_grid(images, row_wise, plot_size=5):
    """
    Create a grid with all images.
    """
    n_images, n_columns, img_fm, img_sz, _ = images.size()
    if not row_wise:
        images = images.transpose(0, 1).contiguous()
    images = images.view(n_images * n_columns, img_fm, img_sz, img_sz)
    images.add_(1).div_(2.0)
    return make_grid(images, nrow=(n_columns if row_wise else n_images))


def compute_grid(interpolations, params):
    # generate the grid / save it to a PNG file
    grid = get_grid(interpolations, params.row_wise, params.plot_size)
    
    attrs =  [int(x) for x in  params.selected_attr.split(',')]
    outdir = 'imgs'
    os.system('mkdir -p %s' % outdir)
    outfname = '%s/tmp.png' % (outdir)
    matplotlib.image.imsave(outfname, grid.numpy().transpose((1, 2, 0)))
    print 'saved', outfname
    return outfname


#################################################################
# MAIN
if __name__ == "__main__":
    run(port=int(params.port))
