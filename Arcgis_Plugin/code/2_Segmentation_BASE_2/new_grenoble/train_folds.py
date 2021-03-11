import argparse, copy, shutil
import os, random
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
#import network_factory as factory
import network_factory_refactor_speed as factory
import dataloader
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import StepLR
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
    parser.add_argument('--use_weight_decay', type=str, required=False, default='False',
                        help='Use weight_decay.')

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
    use_weight_decay = True if args.use_weight_decay == 'True' else False
    
    #TODO: get total of classes from mask
    
    
    print(args)
    
    list_classes = factory.get_classes(inRasterReference, ignore_zero)
    num_classes = len(list_classes)
    
    
    
    #Delete
    base_output = os.path.join(os.getcwd(), out_dir, '2_Segmentation_BASE')
    if os.path.exists(base_output):
         shutil.rmtree(base_output, ignore_errors=True)
         
    #Recreate folders
    os.makedirs(base_output, exist_ok=True)
         
         
    
    #if (not os.path.exists(out_dir)):
    #    os.makedirs(out_dir)

    num_folds = 5
    data = np.genfromtxt('code/config/stratified_folds', dtype=str, delimiter=',')
    for id_fold in range(num_folds):

        print ('.......Creating model.......')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        if modelpath:
            print('Loading model in: {}'.format(modelpath))
            model = torch.load(modelpath)
        else:
            print('Creating a new model: {}'.format(net_type))
            model = factory.model_factory(net_type, num_classes, feature_extract, fine_tunning, isRGB)
        
        # Use multiples GPUS
        #if torch.cuda.device_count() > 1:
        #    print("Let's use", torch.cuda.device_count(), "GPUs!")
        #    batch_size = int(torch.cuda.device_count())*batch_size
        #    model = nn.DataParallel(model, device_ids=[0, 1])
        
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
                                                                  num_workers=4,
                                                                  drop_last=True)
        
        dataset_val = dataloader.RS_Loader_Semantic(dataset_dir, num_classes, mode='val', indices=indices['val'], isRGB=isRGB)
        dataloaders_dict['val'] = torch.utils.data.DataLoader(dataset_val, 
                                                                  batch_size=batch_size, 
                                                                  #sampler=sampler, #torch.utils.data.SubsetRandomSampler(indices[x]), 
                                                                  num_workers=4)
                                                                  
                                                                  
        #Compute mean,std from training data
        mean, std, transform_train = dataloader.get_transforms(dataloaders_dict['train'], 'train', mean=None, std=None)
        dataloaders_dict['train'].dataset.transform = transform_train
        dataloaders_dict['train'].dataset.mean = mean
        dataloaders_dict['train'].dataset.std = std
        dataloaders_dict['train'].dataset.iscomputed = True
        
        _, _, transform_val = dataloader.get_transforms(dataloaders_dict['val'], 'val', mean=mean, std=std)
        dataloaders_dict['val'].dataset.transform = transform_val
        dataloaders_dict['val'].dataset.mean = mean
        dataloaders_dict['val'].dataset.std = std
        
        print ('......Dataloader created......')
        print(dataloaders_dict['train'].dataset.mean)
        print(dataloaders_dict['train'].dataset.std)


        #print(only_top_layers)
        # FOr default all parameters have requires_grad = True
        # So, unselect all layers from backbone if only top is needed
        if only_top_layers == 'True':
            print('TRAINING: ONLY TOP LAYERS')
            # Freeze backbone parameters
            for param in model.backbone.parameters():
                param.requires_grad = False
        else:
            print('TRAINING FULL LAYERS')
            
        # Show trainable layers   
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
                
        # Get parameters to pass to optimizer
        params_to_update = model.parameters()


        """
        print("Params to learn:")
        if feature_extract:
            params_to_update = []
            for name,param in model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t",name)
        else:
            for name,param in model.named_parameters():
                if param.requires_grad == True:
                    print("\t",name)
        """
        
        
        
        #self.weight_class = 1. / np.unique(np.array(self.list_labels), return_counts=True)[1]
        #self.samples_weights = self.weight_class[self. list_labels]
        #criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        #defining optimizer and loss
        scheduler = None
        if opt_type == 'adam':
            
            #Weight_Decay=1e-4# Weight for the weight decay loss function
            #optimizer = optim.Adam(params_to_update, lr=learning_rate, weight_decay=Weight_Decay)
            optimizer = optim.Adam(params_to_update, lr=learning_rate) #//, weight_decay=Weight_Decay)
            
            # Forum Post
            #decayRate = 0.96
            #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
            if use_weight_decay:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        else:
        
            # Optimizer
            optimizer = optim.SGD(params_to_update, lr=learning_rate, momentum=0.9)
            
            # Use lr decay for sgd optimizer
            if use_weight_decay:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
                #lambda1 = lambda epoch: learning_rate / (1. + epoch*1e-7)
                #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
            
            
            
        
        
        
        # for brumadinho need ignore index 0
        if ignore_zero:
            print('Ignoring Index Zero')
            criterion = nn.CrossEntropyLoss(ignore_index=254, weight=torch.from_numpy(dataloaders_dict['train'].dataset.class_weights).float().cuda())
            #criterion = nn.CrossEntropyLoss(ignore_index=254)
        else:
            criterion = nn.CrossEntropyLoss()
        
        
        
        
        
        
        tensor_board = SummaryWriter(log_dir = base_output)
        final_model, val_history = factory.train(model, dataloaders_dict, criterion, optimizer,
                                                 epochs, early_stop, tensor_board, net_type, scheduler, opt_type, list_classes, use_weight_decay)
                                                 
        
        #if fine_tunning:
        #    torch.save(final_model, os.path.join(out_dir, net_type + '_final_model_ft'))
        #    final_stats_file = open (os.path.join(out_dir, net_type + '_finalstats_ft.txt'), 'w')
        #else:
        #    torch.save(final_model, os.path.join(out_dir, net_type + '_final_model'))
        #    final_stats_file = open (os.path.join(out_dir, net_type + '_finalstats.txt'), 'w')
        #factory.final_eval(model, dataloaders_dict, batch_size, final_stats_file)
        
        # Save mean and std
        final_model.mean = copy.deepcopy(mean)
        final_model.std = copy.deepcopy(std)
        final_model.num_classes = copy.deepcopy(num_classes)
        
        torch.save(final_model, os.path.join(base_output, net_type + '_final_model_ft_fold_{}'.format(id_fold)))
        
        #'''
        #final_model = torch.load(os.path.join(out_dir, net_type + '_final_model_ft'))
        # SAVE FINAL MAP
        # TODO: THIS FINAL MAP RASTER NEED TO BE REFERENCED
        #inRaster = '/home/edemir/FAPEMIG_BRUMADINHO/data/montesanto/image.tif'
        
        #inRaster = '../data/brumadinho/Geoeye_Raster/Geoeye_T3.tif'
        #MDT_Raster = '../data/brumadinho/MDT/MDT_T3_Resampled.tif'
        #DEC_Raster = '../data/brumadinho/Declividade/Declividade_T3_Resampled.tif'
        #output_name = os.path.join(out_dir, net_type + '_final_map_T3.tif')
        #factory.create_map(final_model, inRaster, MDT_Raster, DEC_Raster, num_classes, output_name) 
        
        #imsave(os.path.join(out_dir, net_type + '_final_map.png'), map_preds)
        #'''
        
        # Time T2
        
        #inRaster = '../data/brumadinho/Geoeye_Raster/Geoeye_T2.tif'
        #MDT_Raster = '../data/brumadinho/MDT/MDT_T2_Resampled.tif'
        #DEC_Raster = '../data/brumadinho/Declividade/Declividade_T2_Resampled.tif'
        #output_name = os.path.join(out_dir, net_type + '_final_map_T2.tif')
        #factory.create_map(final_model, inRaster, MDT_Raster, DEC_Raster, num_classes, output_name) 
        
        # Time T1
        #inRaster = '../data/brumadinho/Geoeye_Raster/Geoeye_T3.tif'
        #output_name = os.path.join(out_dir, net_type + '_final_map.png')
        #factory.create_map(model, inRaster, MDT_Raster, DEC_Raster, num_classes, output_name) 
    
    
if __name__ == '__main__':
    main()
