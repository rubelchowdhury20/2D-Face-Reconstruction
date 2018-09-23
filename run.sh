tensorboard --logdir=/tmp/SFSNet/logs/ --port=6006 &
python train.py --skipnet_batch_size 1 --skipnet_epochs 1
