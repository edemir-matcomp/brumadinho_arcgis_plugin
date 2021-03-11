import os, time


mydir = os.path.split(os.path.abspath(__file__))[0]
homepath = os.path.expanduser(os.getenv('USERPROFILE'))
#print('Meu diretorio: {}'.format(mydir))
#print('Minha Home: {}'.format(homepath))

list_lib = [r'{}\anaconda3\envs\arc105'.format(homepath), \
     r'{}\anaconda3\envs\arc105\Library\mingw-w64\bin'.format(homepath), \
     r'{}\anaconda3\envs\arc105\Library\usr\bin'.format(homepath), \
     r'{}\anaconda3\envs\arc105\Library\bin'.format(homepath), \
     r'{}\anaconda3\envs\arc105\Scripts'.format(homepath), \
     r'{}\anaconda3\envs\arc105\bin'.format(homepath), \
     r'{}\anaconda3\condabin'.format(homepath)]

for i in list_lib:
    os.environ['PATH'] = '%s;%s' % (i, os.environ['PATH'])

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import argparse, copy
import random
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
#import network_factory as factory
import network_factory_refactor_speed as factory
import dataloader, dataloader_map
import torch.nn as nn
import torch
from skimage.io import imsave

# Fix seed's for reproducibility
random.seed(42)
torch.manual_seed(42)

def main():
    parser = argparse.ArgumentParser(description='Semantic Segmentation General')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to folder where models and stats will be saved')
    parser.add_argument('--model_path', type=str, required=True,
                        help = 'Choose trained model')
    parser.add_argument('--batch', type=int, required=True,
                        help='Batch Size')
                            
    # Inputs and Reference 
    parser.add_argument('--inRaster_Image', type=str, required=True,
                        help='Path to Raster Image')
    parser.add_argument('--inRaster_MDT', type=str, required=False,default='',
                        help='Path to MDT Image')
    parser.add_argument('--inRaster_DEC', type=str, required=False,default='',
                        help='Path to DEC Image')
    parser.add_argument('--inRaster_Reference', type=str, required=False,
                        help='Path to Raster Reference if Exists')
                        
    parser.add_argument('--ignore_zero', type= bool, required=False, default=True,
                        help='Ignore class 0 (background).')

    args = parser.parse_args()
    out_dir = args.output_path
    model_path = args.model_path
    batch_size = args.batch

    inRaster = args.inRaster_Image
    inRaster_MDT = None if args.inRaster_MDT == 'None' else args.inRaster_MDT
    inRaster_DEC = None if args.inRaster_DEC == 'None' else args.inRaster_DEC

    print(args)

    # Create output path if not exists
    base_output = os.path.join(out_dir, '4_Evaluate_Model')
    if not os.path.exists(base_output):
         os.makedirs(base_output, exist_ok=True)
    
    #if (not os.path.exists(out_dir)):
    #    os.makedirs(out_dir)

    print ('.......Creating model.......')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path)
    
    model = model.to(device)
    print(model)
    print ('......Model created.......')
    
    output_name = os.path.join(base_output, os.path.basename(model_path).replace('_model_','_map_'))
    factory.create_map_dataloader(model, inRaster, inRaster_MDT, inRaster_DEC, output_name)
   
if __name__ == '__main__':
    main()
