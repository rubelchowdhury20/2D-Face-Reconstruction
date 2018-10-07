#wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
#bash ~/miniconda.sh -b -p ~/miniconda 
#rm ~/miniconda.sh


#sudo -s
# The above command will prompt password entry.

export PATH=/root/miniconda/bin/:$PATH

conda env create -f environment.yml
source activate 3dface
pip install opencv-contrib-python
apt-get -y update
apt-get -y upgrade
apt install -y libsm6 libxext6 libxrender1

# Installing dlib from github
git clone https://github.com/davisking/dlib.git
apt install -y cmake
cd dlib
!python setup.py install
cd ./..

mkdir -p ./models/latest/ 
mkdir -p /tmp/SFSNet/checkpoints/skipnet_checkpoints
mkdir -p /tmp/SFSNet/checkpoints/sfsnet_checkpoints

python data_loader.py
