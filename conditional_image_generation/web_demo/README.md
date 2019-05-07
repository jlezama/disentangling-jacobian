Web demo for facial attribute manipulation, as shown in the paper
"Overcoming the Disentanglement vs Reconstruction Trade-off via
Jacobian Supervision", J. Lezama, ICLR 2019.


Download pretrained model from here: https://www.dropbox.com/s/gj5rt0cx0ld6qdq/student_with_jacobian.pth?dl=0

Download dataset from here: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

To run web demo:

`python web_server.py --port 8080 --model_path ../models/student_with_jacobian.pth`