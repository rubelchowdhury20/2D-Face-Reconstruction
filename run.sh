bash setup.sh
source activate 3dface
tensorboard --logdir=/tmp/SFSNet/checkpoints/ --port=6006 &
python train.py
