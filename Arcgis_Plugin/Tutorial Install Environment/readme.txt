Follow steps:


1) Download and Install Anaconda
2) Create a new env for python 3.8, called 'arc105' (avoiding base env, since it contains a lot of libs)
3) In anaconda prompt, chage to new env: 

> conda activate 105

and install pytorch using:

> conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

4) There are some remains dependencies to install:
in anaconda environments, search for 

> rasterio
> shapely
> pyshp
> scikit-image
> tensorboard
> scikit-learn
> tqdm
> imgaug


Example call from python2.7 to python3.5

import os
import subprocess
from subprocess import Popen, PIPE

# Get python3 from conda env
#args = shlex.split(command)
#my_subprocess = subprocess.Popen(args)

cmd = 'C:\Users\edemir\\anaconda3\envs\\arc105\python.exe check_python_versions.py'

args = shlex.split(command)
my_subprocess = subprocess.Popen(args)






python.exe -W ignore rasterio_tiles.py --inRaster_Image '../0_CREATE_MASK/data_example/raster/subset_image.tif' --inRaster_Reference '../0_CREATE_MASK/data_example/out_raster/mask.tif' --patchSize 256 --outRaster 'results'

'D:\Arcgis_Plugin\example_data\montesanto\dataset_montesanto'

python.exe -W ignore code/train.py --dataset_path 'D:\Arcgis_Plugin\example_data\montesanto\dataset_montesanto' --output_path 'output' --epochs 100 --learning_rate 0.001 --batch 32 --optimizer_type 'adam' --early_stop 20 --fine_tunning_imagenet 'True' --network_type 'fcn_50'