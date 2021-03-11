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
import numpy as np

# Fix seed's for reproducibility
random.seed(42)
torch.manual_seed(42)

def get_indices(id_fold, data):

    # Instantiate Folds
    list_folds = np.array(range(5))
    
    # Roll in axis
    tmp_set = np.roll(list_folds,id_fold)

    indices_train = []
    indices_val = []
    indices_test = []

    # Train
    for i in tmp_set[:3]:
        indices_train += list(data[data[:,1].astype(int) == i][:,0])
    
    # Val
    for i in tmp_set[3:4]:
        indices_val += list(data[data[:,1].astype(int) == i][:,0])
    
    # Test
    for i in tmp_set[4:]:
        indices_test += list(data[data[:,1].astype(int) == i][:,0])
        
    
    #print(indices_train)
    #print(indices_val)
    #print(indices_test)
    
    indices = {}
    indices['train'] = indices_train
    indices['val'] = indices_val
    indices['test'] = indices_test
    
    return indices
    
def main():
    '''
    parser = argparse.ArgumentParser(description='Semantic Segmentation General')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to dataset')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to folder where models and stats will be saved')
    parser.add_argument('--batch', type=int, required=True,
                        help='Batch Size')
    parser.add_argument('--network_type', type=str, required=True,
                        help = 'Choose network type')                        
    parser.add_argument('--ignore_zero', type= bool, required=False, default=True,
                        help='Ignore class 0 (background).')

    args = parser.parse_args()
    dataset_dir = args.dataset_path
    out_dir = args.output_path
    batch_size = args.batch
    net_type = args.network_type
    ignore_zero = args.ignore_zero
    '''
    
    parser = argparse.ArgumentParser(description='Semantic Segmentation General')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to dataset')
    parser.add_argument('--inRasterReference', type=str, required=True,
                        help='Path to inRasterReference')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to folder where models and stats will be saved')
    parser.add_argument('--batch', type=int, required=True,
                        help='Batch Size')
    parser.add_argument('--epochs', type=int, required=True,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, required=False, default=0.001,
                        help='Learning rate. Default:0.001')
    parser.add_argument('--network_type', type=str, required=True,
                        help = 'Choose network type')
    parser.add_argument('--optimizer_type', type=str, required=False, default='adam',
                        help = 'Optimizer: adam, sgd')
    parser.add_argument('--early_stop', type=int, required=True,
                        help='Number of epochs to activate early stop.')
    parser.add_argument('--fine_tunning_imagenet', type= bool, required=False, default=False,
                        help='set fine tunning on imagenet.')
    parser.add_argument('--feature_extract', type= bool, required=False, default=False,
                        help='Train just the classifier.')
    parser.add_argument('--only_top_layers', type= str, required=False, default='True',
                        help='Train only the top layers (classifier).')
    parser.add_argument('--ignore_zero', type= bool, required=False, default=True,
                        help='Ignore class 0 (background).')
    parser.add_argument('--modelpath', type=str, required=False, default=False,
                        help='Ignore class 0 (background).')
    parser.add_argument('--isRGB', type=str, required=False, default='False',
                        help='Ignore class 0 (background).')

    args = parser.parse_args()
    dataset_dir = args.dataset_path
    inRasterReference = args.inRasterReference
    out_dir = args.output_path
    batch_size = args.batch
    epochs = args.epochs
    learning_rate = args.learning_rate
    net_type = args.network_type
    opt_type = args.optimizer_type
    fine_tunning = args.fine_tunning_imagenet
    early_stop = args.early_stop
    feature_extract = args.feature_extract
    only_top_layers = args.only_top_layers
    ignore_zero = args.ignore_zero
    modelpath = args.modelpath
    isRGB = True if args.isRGB == 'True' else False
    
    print(args)
    
    # Just to Evaluate the best model after load the best model
    #list_classes = factory.get_classes('../0_CREATE_MASK/data_example/out_raster/mask.tif', ignore_zero)
    #num_classes = len(list_classes)
    
    list_classes = factory.get_classes(inRasterReference, ignore_zero)
    num_classes = len(list_classes)
    
    if (not os.path.exists(out_dir)):
        os.makedirs(out_dir)


    num_folds = 5
    data = np.genfromtxt('code/config/stratified_folds', dtype=str, delimiter=',')
    final_stats_file = open (os.path.join(out_dir, net_type + '_finalstats.txt'), 'w')
    for id_fold in range(num_folds):

        print ('.......Creating model.......')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(os.path.join(out_dir, net_type + '_final_model_ft'+ '_fold_' + str(id_fold)))
        #model = torch.load(model_path)
        
        model = model.to(device)
        print(model)
        print ('......Model created.......')
        
        indices = get_indices(id_fold, data)
        
        print ('......Creating dataloader......')
        dataloaders_dict = {}
        dataset = dataloader.RS_Loader_Semantic(dataset_dir, num_classes, mode='train', indices=indices['train'], isRGB=isRGB)
        dataloaders_dict['train'] = torch.utils.data.DataLoader(dataset, 
                                                                  batch_size=batch_size, 
                                                                  #sampler=sampler, #torch.utils.data.SubsetRandomSampler(indices[x]), 
                                                                  num_workers=4)
        
        dataset_val = dataloader.RS_Loader_Semantic(dataset_dir, num_classes, mode='val', indices=indices['val'], isRGB=isRGB)
        dataloaders_dict['val'] = torch.utils.data.DataLoader(dataset_val, 
                                                                  batch_size=batch_size, 
                                                                  #sampler=sampler, #torch.utils.data.SubsetRandomSampler(indices[x]), 
                                                                  num_workers=4)
    
        dataset_test = dataloader.RS_Loader_Semantic(dataset_dir, num_classes, mode='val', indices=indices['test'], isRGB=isRGB)
        dataloaders_dict['test'] = torch.utils.data.DataLoader(dataset_test, 
                                                                  batch_size=batch_size, 
                                                                  #sampler=sampler, #torch.utils.data.SubsetRandomSampler(indices[x]), 
                                                                  num_workers=4)                                                              
                                                                  
                                                                  
        #Compute mean,std from training data
        #mean, std, transform_train = dataloader.get_transforms(dataloaders_dict['train'], 'train', mean=None, std=None)
        #dataloaders_dict['train'].dataset.transform = transform_train
        #dataloaders_dict['train'].dataset.mean = model.mean
        #dataloaders_dict['train'].dataset.std = model.std
        
        _, _, transform_val = dataloader.get_transforms(dataloaders_dict['val'], 'val', mean=model.mean, std=model.std)
        dataloaders_dict['val'].dataset.transform = transform_val
        dataloaders_dict['val'].dataset.mean = model.mean
        dataloaders_dict['val'].dataset.std = model.std
        
        _, _, transform_test = dataloader.get_transforms(dataloaders_dict['test'], 'val', mean=model.mean, std=model.std)
        dataloaders_dict['test'].dataset.transform = transform_test
        dataloaders_dict['test'].dataset.mean = model.mean
        dataloaders_dict['test'].dataset.std = model.std
        
        
        
        print ('......Dataloader created......')
        final_stats_file.write('Fold {}\n'.format(id_fold))
        #Evaluate in Validation
        factory.final_eval(model, dataloaders_dict, batch_size, final_stats_file, list_classes,mode='val')
        #Evaluate in Test
        factory.final_eval(model, dataloaders_dict, batch_size, final_stats_file, list_classes,mode='test')
    final_stats_file.close()

    '''
    inRaster = '../data/brumadinho/Geoeye_Raster/Geoeye_T3.tif'
    MDT_Raster = '../data/brumadinho/MDT/MDT_T3_Resampled.tif'
    DEC_Raster = '../data/brumadinho/Declividade/Declividade_T3_Resampled.tif'
    output_name = os.path.join(out_dir, net_type + '_final_map_T3.tif')
    factory.create_map(model, inRaster, MDT_Raster, DEC_Raster, num_classes, output_name)
    #imsave(os.path.join(out_dir, net_type + '_final_map.png'), map_preds)
    
    
    
    # Time T2
    inRaster = '../data/brumadinho/Geoeye_Raster/Geoeye_T2.tif'
    MDT_Raster = '../data/brumadinho/MDT/MDT_T2_Resampled.tif'
    DEC_Raster = '../data/brumadinho/Declividade/Declividade_T2_Resampled.tif'
    output_name = os.path.join(out_dir, net_type + '_final_map_T2.tif')
    factory.create_map(model, inRaster, MDT_Raster, DEC_Raster, num_classes, output_name)
    
    
    # Time T1
    #inRaster = '../data/brumadinho/Geoeye_Raster/Geoeye_T3.tif'
    #output_name = os.path.join(out_dir, net_type + '_final_map.png')
    #factory.create_map(model, inRaster, num_classes, output_name) 
    '''
    
if __name__ == '__main__':
    main()
