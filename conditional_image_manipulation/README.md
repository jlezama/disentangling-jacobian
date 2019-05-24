# "Overcoming the Disentanglement vs Reconstruction Trade-off via Jacobian Supervision"

This repository contains the source code for the experiments of the article

    "Overcoming the Disentanglement vs Reconstruction Trade-off via Jacobian Supervision"
    José Lezama, ICLR 2019

[https://openreview.net/pdf?id=Hkg4W2AcFm](https://openreview.net/pdf?id=Hkg4W2AcFm)

If you find this work useful in your research, please consider citing this paper.


# Conditional Image Manipulation Training


## Dataset
* Download the aligned and cropped CelebA images and annotations from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html.
* Preprocess using ```preprocess.py``` to create a pth file with all images.


## Training
* To train the _teacher_, set the parameter `lambda_jacobian` to `0`. For example:
```
python train.py  --debug False --n_layers 7 --attr '*'  --lambda_ae 500 --n_epochs 5 --epoch_size 1000 --lambda_ttributes 1 --batch_size 64 --lambda_xcov 0.01 --lambda_flipped 0 --lambda_latent_match 0 --outdir training_test --freeze_encoder 0 --freeze_encoder_step2 1 --clip_grad_norm 10 --max_fm 512 --lambda_jacobian 0  --lambda_y 0 
```
* The parameter `max_fm` controls the depth of the encoder output (which is convolutional). The latent dimension is the spatial resolution times the depth of this feature map (e.g. 4*512 if `n_layers` is 7.

* To train the _student_, set the parameter `lambda_jacobian` to a positive value, and use `ae_reload` to point to the file containing the teacher model weights. For example:
```
python train.py  --debug False --n_layers 7 --attr '*'  --lambda_ae 500 --n_epochs 5 --epoch_size 1000  --lambda_ttributes 0 --batch_size 64 --lambda_xcov 0.01 --lambda_flipped 1e-3 --lambda_latent_match 1e-4 --outdir training_test_student --freeze_encoder 0 --freeze_encoder_step2 1 --clip_grad_norm 10 --max_fm 1024 --lambda_jacobian 100  --lambda_y 1 --ae_reload /data/tmp/training_test/epoch_0000.pth_ae.pth
```
* Use `--debug True` to train and test on a small subset of the training set.
* The file `train_examples.sh` includes training examples.
* A pre-trained model can be found [here] (https://www.dropbox.com/s/gj5rt0cx0ld6qdq/student_with_jacobian.pth?dl=0)

## Web Interface
The source code for creating an interactive manipulation interface is found in [web_demo](web_demo)

