import argparse, copy
import os, random
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
#import network_factory as factory
import network_factory_refactor_speed as factory
import dataloader
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
    parser.add_argument('--inRaster_MDT', type=str, required=True,
                        help='Path to MDT Image')
    parser.add_argument('--inRaster_DEC', type=str, required=True,
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
    inRaster_DEC = args.inRaster_DEC
    inRaster_MDT = args.inRaster_MDT
    
    #TODO: get total of classes from mask
    
    print(args)
    
    #list_classes = factory.get_classes('../0_CREATE_MASK/data_example/out_raster/mask.tif', ignore_zero)
    #num_classes = len(list_classes)
    
    list_classes = list(range(12)) # [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13.]
    num_classes = len(list_classes)
    
    if (not os.path.exists(out_dir)):
        os.makedirs(out_dir)

    print ('.......Creating model.......')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path)
    
    model = model.to(device)
    print(model)
    print ('......Model created.......')
    
    #'''
    #inRaster = '../data/brumadinho/Geoeye_Raster/Geoeye_T3.tif'
    #MDT_Raster = '../data/brumadinho/MDT/MDT_T3_Resampled.tif'
    #DEC_Raster = '../data/brumadinho/Declividade/Declividade_T3_Resampled.tif'
    
    model_path
    output_name = os.path.join(out_dir, os.path.basename(model_path).replace('_model_','_map_'))
    factory.create_map(model, inRaster, inRaster_MDT, inRaster_DEC, num_classes, output_name)
    #imsave(os.path.join(out_dir, net_type + '_final_map.png'), map_preds)
    #'''
    
    #'''
    # Time T2
    #inRaster = '../data/brumadinho/Geoeye_Raster/Geoeye_T2.tif'
    #MDT_Raster = '../data/brumadinho/MDT/MDT_T2_Resampled.tif'
    #DEC_Raster = '../data/brumadinho/Declividade/Declividade_T2_Resampled.tif'
    #output_name = os.path.join(out_dir, net_type + '_final_map_T2.tif')
    #factory.create_map(model, inRaster, MDT_Raster, DEC_Raster, num_classes, output_name)
    #'''
    
    # Time T1
    #inRaster = '../data/brumadinho/Geoeye_Raster/Geoeye_T3.tif'
    #output_name = os.path.join(out_dir, net_type + '_final_map.png')
    #factory.create_map(model, inRaster, num_classes, output_name) 
    
    
if __name__ == '__main__':
    main()
