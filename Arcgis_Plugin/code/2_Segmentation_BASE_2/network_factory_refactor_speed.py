import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import models
import statistics
import time
import copy, os
from tqdm import tqdm
import dataloader_map

from skimage.exposure import match_histograms
from skimage.io import imread, imsave

from sklearn.metrics import confusion_matrix

import rasterio
from rasterio.windows import Window
from torchvision import transforms

#FCN with ResNet-50/ResNet-101 backbone
#DeepLab-V3 with ResNet-50/ResNet-101 backbone

#torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True, num_classes=21, aux_loss=None, **kwargs)
#torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, progress=True, num_classes=21, aux_loss=None, **kwargs)

def confusion_matrix_1(y_true, y_pred, labels):
    N = len(labels) #max(max(y_true), max(y_pred)) + 1
    y_true = torch.tensor(y_true, dtype=torch.long)
    y_pred = torch.tensor(y_pred, dtype=torch.long)
    return torch.sparse.LongTensor(
        torch.stack([y_true, y_pred]), 
        torch.ones_like(y_true, dtype=torch.long),
        torch.Size([N, N])).to_dense().numpy()
        
def confusion_matrix_2(y_true, y_pred, labels):
    N = len(labels) #max(max(y_true), max(y_pred)) + 1
    y_true = torch.tensor(y_true, dtype=torch.long)
    y_pred = torch.tensor(y_pred, dtype=torch.long)
    y = N * y_true + y_pred
    y = torch.bincount(y)
    if len(y) < N * N:
        y = torch.cat(y, torch.zeros(N * N - len(y), dtype=torch.long))
    y = y.reshape(N, N)
    return y.numpy()

def get_classes(InRaster_Reference, ignore_zero):

    # Get number of classes
    mask = imread(InRaster_Reference)
    list_classes = np.unique(mask)
    if ignore_zero:
        list_classes = list_classes[:-1]
        
    print('List of {} classes from Reference Raster: {}'.format(len(list_classes), list_classes))

    return list_classes

def model_factory(model_name, num_classes, feature_extract=False, use_pretrained=True, isRGB=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    #input_size = 0
    
    
    if isRGB:
        num_channels=3
    else:
        num_channels=5

    print('Trying load mode: {}'.format(model_name))

    if model_name == "fcn_50":
        """ FCN with ResNet-50 backbone
        """
        # DONT HAVE A PRETRAINED MODEL
        model_ft = models.segmentation.fcn_resnet50(pretrained=False, progress=True, num_classes=num_classes, aux_loss=None)
        
        # Change input layer to receive 6 bands
        model_ft.backbone.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
        # Change output layer to receive 'num_classes'
        model_ft.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))

    elif model_name == "fcn_101":
        """ FCN with ResNet-101 backbone
        """
        if use_pretrained:
            model_ft = models.segmentation.fcn_resnet101(pretrained=True, progress=True, num_classes=21, aux_loss=None)
        else:
            model_ft = models.segmentation.fcn_resnet101(pretrained=False, progress=True, num_classes=21, aux_loss=None)
        #model_ft = models.segmentation.fcn_resnet101(pretrained=True, progress=True, num_classes=21, aux_loss=None)
        #print(model_ft)
        #exit(1)
        
        # Change input layer to receive 6 bands
        model_ft.backbone.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Change output layer to receive 'num_classes'
        model_ft.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        
             
    elif model_name == "deeplabv3_50":
        """ DeepLab-V3 with ResNet-50 backbone
        """
        
        model_ft = models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True, num_classes=num_classes, aux_loss=None)
        
        # Change input layer to receive 6 bands
        model_ft.backbone.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Change output layer to receive 'num_classes'
        model_ft.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        
        

    elif model_name == "deeplabv3_101":
        """ DeepLab-V3 with ResNet-101 backbone
        """

        if use_pretrained:
            model_ft = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True, num_classes=21, aux_loss=None)
        else:
            model_ft = models.segmentation.deeplabv3_resnet101(pretrained=False, progress=True, num_classes=21, aux_loss=None)
        
        # Change input layer to receive 6 bands
        model_ft.backbone.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Change output layer to receive 'num_classes'
        model_ft.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        
    else:
        print("Invalid model name, exiting...")
        exit()
        
    return model_ft


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def train(model, dataloaders, criterion, optimizer, num_epochs, epochs_early_stop, tensor_board, model_name, scheduler, opt_type, list_classes, use_weight_decay):

    num_classes = dataloaders['train'].dataset.num_classes
    counter_early_stop_epochs = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    since = time.time()
    #counter = 0
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf
    best_acc = 0.0
    best_epoch = 0
    stride = 64
    size = 128

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('Learning Rate {}'.format(optimizer.param_groups[0]['lr']))
        print('-' * 10)
        predictions = []
        labels_list = []

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
        #for phase in ['val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            cm_total = np.zeros((num_classes, num_classes))

            # Iterate over data.
            for inputs, labels, inputs_path in tqdm(dataloaders[phase]):
            
                tx = time.time()
                #print(inputs.shape)
                #print(inputs[0].shape)
                #imsave('teste.png', np.transpose(inputs[0],(1,2,0)))
                #imsave('teste_labels.png', labels[0])
                #exit(1)
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # for brumadinho need ignore index 254
                # mask_pixels = (labels.data != 254).cpu().numpy()
                
                #print('Imgs Shape', inputs.shape)
                #print('Mask Shape', labels.shape)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    
                    # Worth create a function for val phase?
                    # Get a bigger image(256,256), forward in all windows (128,128) with stride 64 
                    # and recompile image to output 256,256
                     
                    outputs = model(inputs)['out']
                    loss = criterion(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)
                          
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                        # Every 5 epochs, decay lr
                        
                        #if epoch % 5 == 0 and epoch != 0 and use_weight_decay:
                        #    scheduler.step()
                    
                    
                    
               

                # statistics
                #preds_tmp = preds.data.cpu().numpy().copy()
                #labels_tmp = labels.data.cpu().numpy().copy()
                
                # for brumadinho
                # preds_tmp = preds_tmp[mask_pixels]
                # labels_tmp = labels_tmp[mask_pixels]
                
                #for p in preds_tmp:
                #    predictions.append(p.ravel())
                #for l in labels_tmp: 
                #    labels_list.append(l.ravel())
                
                running_loss += loss.item() * inputs.size(0)
            

                #print(type(labels_tmp))
                #print(type(preds_tmp))
                
                #print(labels_tmp.shape)
                #print(preds_tmp.shape)
                
                #for brumadinho
                #cm_train = confusion_matrix(labels_tmp.ravel(), preds_tmp.ravel(), labels=list_classes)
                
                t0 = time.time()
                
                #l1 = labels.detach().cpu().numpy()
                #p1 = preds.detach().cpu().numpy()
                
                #print("l1 shape {}".format(l1.shape))
                #print("p1 shape {}".format(p1.shape))
                
                #cm_train = confusion_matrix(labels.view(-1).detach(), preds.view(-1).detach(), labels=list_classes)
                
                cm_train = confusion_matrix_1(labels.view(-1), preds.view(-1), labels=list_classes)
                #cm_train = confusion_matrix_2(labels.view(-1), preds.view(-1), labels=list_classes)
                
                #coffee
                #cm_train = confusion_matrix(labels_tmp.ravel(), preds_tmp.ravel(), labels=[0,1])
                cm_total += cm_train
                #print(cm_total)
                
                mean_acc = statistics.balanced_accuracy_score_from_cm(cm_train)
                f1_score = statistics.f1_score_from_cm(cm_train)
                
                #exit(1)
                #print('Time Compute Matrix and detach to numpy {}'.format(time.time()-t0))
                
                #print('Total time {}'.format(time.time()-tx))
                #exit(1)
               
                ''' '''
                
            print(cm_total)
            epoch_loss = running_loss / dataloaders[phase].dataset.len
            epoch_acc = statistics.balanced_accuracy_score_from_cm(cm_total)
            epoch_f1 = statistics.f1_score_from_cm(cm_total)
            

            print('Epoch: {} Phase: {} Loss: {:.4f} Acc: {:.4f}'.format(epoch, phase, epoch_loss, epoch_acc))
            if (phase == 'train'):
                tensor_board.add_scalar('Loss/train', epoch_loss, epoch)
                tensor_board.add_scalar('Accuracy/train', epoch_acc, epoch)
            else:
                tensor_board.add_scalar('Loss/val', epoch_loss, epoch)
                tensor_board.add_scalar('Accuracy/val', epoch_acc, epoch)

            # deep copy the model
            if phase == 'val':
                # For ReduceLROnPlateau scheduler
                scheduler.step(epoch_loss)

                counter_early_stop_epochs += 1
                val_acc_history.append(epoch_acc)
            #if phase == 'val' and epoch_acc => best_acc:
            if phase == 'val' and epoch_loss < best_loss: 
                counter_early_stop_epochs = 0
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
                print('New Best - Epoch: {} Phase: {} Loss: {:.4f} Acc: {:.4f}'.format(epoch, phase, epoch_loss, epoch_acc))
                       
            #print(np.asarray(predictions).shape)
            #print(np.asarray(predictions).shape)
            #print(np.asarray(labels_list).shape)
            
            # TODO SLOW OPERATION
            #statistics.calculate_metrics(np.asarray(predictions).ravel(), np.asarray(labels_list).ravel(), list_labels=[0,1])
            
            #predictions = []
            #labels_list = []
        
        if (counter_early_stop_epochs >= epochs_early_stop):
            print ('Stopping training because validation loss did not improve in ' + str(epochs_early_stop) + ' consecutive epochs.')
            break
        
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}, Epoch: {}'.format(best_acc, best_epoch))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def final_eval(model, dataloaders, batch, file, list_classes, mode='val'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    predictions = []
    labels_list = []
    num_classes = len(list_classes)
    cm_total = np.zeros((num_classes, num_classes))
    for inputs, labels, inputs_path in dataloaders[mode]:
        inputs = inputs.to(device)
        labels = labels.to(device)
        mask_pixels = (labels.data != 254).cpu().numpy()
        with torch.set_grad_enabled(False):
            outputs = model(inputs)['out']
            _, preds = torch.max(outputs, 1)
            
        # statistics
        preds_tmp = preds.data.cpu().numpy().copy()
        labels_tmp = labels.data.cpu().numpy().copy()
        
        # for brumadinho
        preds_tmp = preds_tmp[mask_pixels]
        labels_tmp = labels_tmp[mask_pixels]
        
        for p in preds_tmp:
            predictions.append(p.ravel())
        for l in labels_tmp: 
            labels_list.append(l.ravel())
            
        '''
        for c in [3,6,9,13]:
            try:
                labels_tmp[labels_tmp == c] = 0
                preds_tmp[preds_tmp == c] = 0
            except:
                print('Exception in class {}'.format(c))
                continue 
        '''
        
        cm_train = confusion_matrix(labels_tmp.ravel(), preds_tmp.ravel(), labels=list_classes)
        cm_total += cm_train
        
            
    epoch_acc = statistics.balanced_accuracy_score_from_cm(cm_total)
    epoch_f1 = statistics.f1_score_from_cm(cm_total)
    epoch_kappa = statistics.cohen_kappa_from_cm(cm_total, weights=None)
    
    print (cm_total)
    print ("Balanced_Accuracy: " + str(epoch_acc))
    print ("Kappa: " + str(epoch_kappa))
    print ("F1: " + str(epoch_f1))
    
    
    
    file.write("Balanced_Accuracy: " + str(epoch_acc) + "\n")
    file.write("Kappa: " + str(epoch_kappa) + "\n")
    file.write("F1: " + str(epoch_f1) + "\n")
    #file.close()
     
    #Computa tudo de uma vez, mas em seg semantic pode n ter memoria para isso       
    #statistics.calculate_metrics(predictions, labels_list, file, list_classes)
    
def create_map(model, inRaster, MDT_Raster, DEC_Raster, num_classes, output_name, size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    model = model.to(device)
    model.eval()
    predictions = []
    labels_list = []
    num_classes = model.num_classes
    
    isRGB = None
    #if 'RGB' in output_name:
    if model.backbone.conv1.in_channels == 3:
        isRGB=True
    else:
        isRGB=False
    
    stride = int(size/2)
    with rasterio.open(inRaster) as src:
    
        # Get Info from original raster
        image = src.read()[:,:,:]
        bands,rows,cols = image.shape
        
        if not isRGB:
            MDT_Raster = rasterio.open(MDT_Raster).read()[:,:,:]
            DEC_Raster = rasterio.open(DEC_Raster).read()[:,:,:]
        
        #print(image.shape)
        #print(MDT_Raster.shape)
        #print(DEC_Raster.shape)
        
        
        #total_w = int(math.ceil(cols/size))
        #total_h = int(math.ceil(rows/size))

        #TODO: KEEP IN TENSOR TO BE FASTER
        #final_map = np.zeros((2, rows, cols))
        
        final_map = torch.from_numpy(np.zeros((num_classes, rows, cols)))
    
        for i in tqdm(range(0,rows,stride)):
            for j in range(0,cols,stride):
            
                # Rasterio window crop
                start_i, start_j = i, j
                if i+size >= rows:
                    start_i = rows - size
                    
                if j+size >= cols:
                    start_j = cols - size
                
                t0 = time.time()
                window_crop = Window.from_slices((start_i, min(start_i + size, rows)), 
                                                 (start_j, min(start_j + size, cols)))
                #print('Window crop {}'.format(time.time()-t0))              
                
                # TODO: Try to load from the entire image
                #crop = np.transpose(src.read(window=window_crop), (1,2,0))
                
                t0 = time.time()
                crop = np.transpose(image[:, start_i:min(start_i+size,rows) , start_j:min(start_j+size,cols)], (1,2,0))
                
                
                # Selecting only NIR R G
                crop = crop[:,:,[3,0,1]]
                #print('Crop Image crop {}'.format(time.time()-t0))
                
                
                '''
                Load for MDT AND DEC
                '''
                if not isRGB:
                    crop_mdt = np.transpose(MDT_Raster[:, start_i:min(start_i+size,rows) , start_j:min(start_j+size,cols)], (1,2,0))
                    crop_dec = np.transpose(DEC_Raster[:, start_i:min(start_i+size,rows) , start_j:min(start_j+size,cols)], (1,2,0))
                    crop = np.concatenate((crop, crop_dec, crop_mdt), axis=2)
                    
                    #print(crop.shape)
                    #print(crop_mdt.shape)
                    #print(crop_dec.shape)
                             
                t0 = time.time()
 
                # Preprocessing crop
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(model.mean, model.std)])
                             
                inputs = transform(crop)                
                inputs = torch.unsqueeze(inputs, 0).float().to(device)
                #print('Preprocess crop {}'.format(time.time()-t0))

                #print('{} {} {} {} {}'.format(i,j,inputs.shape,rows,cols))
                #print('{} {}'.format(start_i,start_j))
                
                
                with torch.set_grad_enabled(False):
                    t0 = time.time()
                    outputs = model(inputs)['out']
                    #print('Forward crop {}'.format(time.time()-t0))
                    
                    #TODO: KEEP IN TENSOR TO FASTER
                    #final_map[:, 
                    #          start_i:min(start_i + size, rows),
                    #          start_j:min(start_j + size, cols)] += outputs[0].cpu().numpy()
                    t0 = time.time()
                    final_map[:, 
                              start_i:min(start_i + size, rows),
                              start_j:min(start_j + size, cols)] += outputs[0].cpu()
                    #print('Sum Final Map {}'.format(time.time()-t0))
                    
                    #_, preds = torch.max(outputs, 1)
                    
        #TODO: KEEP IN TENSOR TO FASTER
        #preds = np.argmax(final_map, axis=0)
        _, preds = torch.max(final_map, 0)
        
        
        # Save new image georeferenced tif
        preds = preds.cpu().numpy().astype("float32")
        bin_mask_meta = src.meta.copy()
        bin_mask_meta.update({'count': 1})

        with rasterio.open(output_name+".tif", 'w', **bin_mask_meta) as dst:
            dst.write(preds, 1)
        
        print("Save in {}".format(output_name))
        
def create_map_dataloader(model, inRaster, MDT_Raster, DEC_Raster, output_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    model = model.to(device)
    model.eval()
    
    isRGB = None
    if model.backbone.conv1.in_channels == 3:
        isRGB=True
    else:
        isRGB=False

    dataset_path = {'NEW_Raster': inRaster ,
                    'MDT_Raster': MDT_Raster ,
                    'DEC_Raster': DEC_Raster}

    dataset = dataloader_map.RS_Loader_Semantic_Map(dataset_path, model.mean, model.std, isRGB)

    loader = torch.utils.data.DataLoader(dataset, 
                                         batch_size=4, 
                                         num_workers=0,
                                         shuffle=False
                                         )

    # Keep entire final map
    rows, cols = loader.dataset.rows, loader.dataset.cols
    final_map = torch.from_numpy(np.zeros((model.num_classes, rows, cols)))
    size = loader.dataset.size
    
    with torch.set_grad_enabled(False):

        for inputs, start_i, start_j in tqdm(loader):
            #print("New Batch")
            #print("Memory alloc GPU 1", torch.cuda.memory_allocated(0))
            inputs = inputs.to(device)
            #print("Memory alloc GPU 2", torch.cuda.memory_allocated(0))
            #print("inputs",inputs.size())
            outputs = model(inputs)['out'].detach().cpu()
            #print("Memory alloc GPU 3", torch.cuda.memory_allocated(0))
            #print("outputs",outputs.size())

            for index, out in enumerate(outputs):
                start_i_tmp = start_i[index]
                start_j_tmp = start_j[index]
                #print("start_i_tmp",start_i_tmp)
                #print("start_j_tmp",start_j_tmp)
                #print("out size",out.size())
                final_map[:, 
                            start_i_tmp:min(start_i_tmp + size, rows),
                            start_j_tmp:min(start_j_tmp + size, cols)] += out.detach().cpu()

            #print("Memory alloc GPU 4", torch.cuda.memory_allocated(0))
                    
    _, preds = torch.max(final_map, 0)
        
        
    # Save new image georeferenced tif
    preds = preds.cpu().numpy().astype("float32")
    bin_mask_meta = dataset.metadata
    bin_mask_meta.update({'count': 1})

    with rasterio.open(output_name+".tif", 'w', **bin_mask_meta) as dst:
        dst.write(preds, 1)
        
    print("Save in {}".format(output_name))
