wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda 
rm ~/miniconda.sh

echo "PATH=/root/miniconda/bin/:$PATH" >> ~/.bashrc
source ~/.bashrc

conda env create -f environment.yml
source activate 3dface
pip install opencv-contrib-python
apt-get update
apt-get upgrade
apt install -y libsm6 libxext6 libxrender1

mkdir -p ./models/latest/ 
sudo mkdir -p /tmp/SFSNet/checkpoints/skipnet_checkpoints
sudo mkdir -p /tmp/SFSNet/checkpoints/sfsnet_checkpoints

python data_loader.py --skipnet_batch_size 10
