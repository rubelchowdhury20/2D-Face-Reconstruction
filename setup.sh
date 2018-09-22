conda env create -f environment.yml
source activate 3dface

python data_loader.py --skipnet_batch_size 10