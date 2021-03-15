from torchvision import datasets, transforms
from torch.utils import data
import torchvision
import os, random
import torch
import sys
import numpy as np
import tqdm
from PIL import ImageFile
from skimage.io import imread, imsave

import skimage.transform as scikit_transform
from scipy.ndimage.interpolation import rotate
from scipy.ndimage import zoom
import imgaug.augmenters as iaa
from skimage.util import random_noise
from skimage.util import img_as_ubyte
from skimage import exposure
import rasterio

ImageFile.LOAD_TRUNCATED_IMAGES = True

class RS_Loader_Semantic_Map(data.Dataset):
    def __init__(self, dataset_path, mean, std, isRGB):
        
        self.isRGB = isRGB
        # Keep list of positions i,j of image crops
        self.list_pos = []

        src = rasterio.open(dataset_path['NEW_Raster'])
        self.metadata = src.meta.copy()
        
        # B,H,W -> H,W,B
        self.image = np.transpose(src.read()[:,:,:], (1,2,0))
        self.rows,self.cols,self.bands = self.image.shape
        print("self.image.shape",self.bands,self.rows,self.cols)
        if not self.isRGB:
            self.MDT_Raster = rasterio.open(dataset_path['MDT_Raster']).read()[:,:,:]
            self.DEC_Raster = rasterio.open(dataset_path['DEC_Raster']).read()[:,:,:]


        # List all Window Crops
        self.size = 128
        stride = int(self.size/2)
        for i in range(0,self.rows,stride):
            for j in range(0,self.cols,stride):
            
                # Rasterio window crop
                start_i, start_j = i, j
                if i+self.size >= self.rows:
                    start_i = self.rows - self.size
                    
                if j+self.size >= self.cols:
                    start_j = self.cols - self.size

                self.list_pos.append((start_i,start_j))

        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean, std)])

        self.len = len(self.list_pos)

 
    def __len__(self):
        return self.len

    def __getitem__(self, index):

        # Get new image/mask path
        start_i, start_j = self.list_pos[index]

        # Crop from original image
        crop = self.image[start_i:min(start_i+self.size,self.rows) , start_j:min(start_j+self.size,self.cols), :].copy()

        # Selecting only NIR R G
        crop = crop[:,:,[3,0,1]].copy()

        '''
        Load for MDT AND DEC
        '''
        if not self.isRGB:
            crop_mdt = np.transpose(self.MDT_Raster[:, start_i:min(start_i+self.size,self.rows) , start_j:min(start_j+self.size,self.cols)], (1,2,0))
            crop_dec = np.transpose(self.DEC_Raster[:, start_i:min(start_i+self.size,self.rows) , start_j:min(start_j+self.size,self.cols)], (1,2,0))
            crop = np.concatenate((crop, crop_dec, crop_mdt), axis=2)

        # Apply transform 
        if self.transform:
            img = self.transform(crop).float()

        #print(img.shape)
        #print(start_i)
        #print(start_j)
        
        return img, start_i, start_j
