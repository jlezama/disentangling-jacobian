# This script can be used to try the training of the teacher and
# student models.
# 
# This training schedule should replicate the experiment in the paper
# by doing the following modifications:
# - setting n_epochs to 100, 100,  200, 200 for A, B, C and D respectively
# - setting epoch_size to the dataset size (160K)
# - setting the ae_reload paths to the previous model at each stage
# - changing --debug True to --debug False

# A) train teacher first as standard autoencoder (d=512*4) for 100 epochs
CUDA_VISIBLE_DEVICES=0 python train.py  --debug True --n_layers 7 --attr '*'  --lambda_ae 100 --n_epochs 1 --epoch_size 1000 --lambda_ttributes 0.1 --batch_size 64 --lambda_xcov 0.1 --lambda_flipped 0 --lambda_latent_match 0 --outdir training_example --freeze_encoder 0 --freeze_encoder_step2 1 --clip_grad_norm 10 --max_fm 512 --lambda_jacobian 0  --lambda_y 0 

# B) fine-tune the teacher with cycle-loss for another  100 epochs (see section 4.2.1 in the paper)
CUDA_VISIBLE_DEVICES=0 python train.py  --debug True --n_layers 7 --attr '*'  --lambda_ae 100 --n_epochs 1 --epoch_size 1000 --lambda_ttributes 0.1 --batch_size 64 --lambda_xcov 0.1 --lambda_flipped 1e-4 --lambda_latent_match 1e-5 --outdir training_example --freeze_encoder 0 --freeze_encoder_step2 1 --clip_grad_norm 10 --max_fm 512 --lambda_jacobian 0  --lambda_y 0 --ae_reload /data/tmp/training_example/epoch_0000.pth_ae.pth 


# C) train student (d=1024*4) for 200 epochs (note ae_reload model will be used both for weight initialization and as the teacher)
CUDA_VISIBLE_DEVICES=0 python train.py  --debug True --n_layers 7 --attr '*'  --lambda_ae 100 --n_epochs 1 --epoch_size 1000  --lambda_ttributes 0 --batch_size 64 --lambda_xcov 0.01 --lambda_flipped 1e-3 --lambda_latent_match 1e-4 --outdir training_example_student --freeze_encoder 0 --freeze_encoder_step2 1 --clip_grad_norm 10 --max_fm 1024 --lambda_jacobian 50  --lambda_y 1 --ae_reload /data/tmp/training_example/epoch_0000.pth_ae.pth

# D) train student (d=2048*4) for 200 epochs (note that in this case the initialization and the teacher model are different)
CUDA_VISIBLE_DEVICES=0 python train.py  --debug True --n_layers 7 --attr '*'  --lambda_ae 100 --n_epochs 1 --epoch_size 1000  --lambda_ttributes 0 --batch_size 64 --lambda_xcov 0.01 --lambda_flipped 1e-3 --lambda_latent_match 1e-4 --outdir training_example_student_final --freeze_encoder 0 --freeze_encoder_step2 1 --clip_grad_norm 10 --max_fm 2048 --lambda_jacobian 50  --lambda_y 1 --ae_reload /data/tmp/training_example_student/epoch_0000.pth_ae.pth --ae_teacher_reload /data/tmp/training_example/epoch_0000.pth_ae.pth
