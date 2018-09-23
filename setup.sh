bash Miniconda3-latest-Linux-x86_64.sh
conda env create -f environment.yml
source activate 3dface

mkdir -p ./models/latest/ 
sudo mkdir -p /tmp/SFSNet/checkpoints/skipnet_checkpoints
sudo mkdir -p /tmp/SFSNet/checkpoints/sfsnet_checkpoints

python data_loader.py --skipnet_batch_size 10
