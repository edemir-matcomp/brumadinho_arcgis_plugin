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

ImageFile.LOAD_TRUNCATED_IMAGES = True

'''
class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the _getitem_ method. this is the method that dataloader calls
    def _getitem_(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self)._getitem_(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        print('OLHA: ' ,self.imgs[index])
        sys.exit()
        return tuple_with_path

def create_dataloader(data_dir, input_size, batch_size):
    data_transforms = {
    'train': transforms.Compose([
        #transforms.RandomResizedCrop(input_size),
        torchvision.transforms.RandomCrop((32,32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        #transforms.Resize(input_size),
        #transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), 
    	              data_transforms[x]) for x in ['train', 'val']}
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                        batch_size=batch_size, shuffle=True, num_workers=4) 
                        for x in ['train', 'val']}
    return dataloaders_dict
'''
def random_crop(img, mask, width, height):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    assert img.shape[0] == mask.shape[0]
    assert img.shape[1] == mask.shape[1]
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    mask = mask[y:y+height, x:x+width]
    return img, mask
    
def recursive_glob(rootdir=".", suffix=".tif"):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot,filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]
    
class RS_Loader_Semantic(data.Dataset):
    def __init__(self, dataset_path, num_classes, mode, indices, isRGB, transform=None, memory_load=True):
    
    
    
        ''' Old Loader - Nao consigo usar esse loader porque dividiria o dataset 5 vezes ''' 
        '''
        # Get images path and labels
        list_images = recursive_glob(os.path.join(dataset_path, mode,'imgs'))
    
        # Get images path and labels
        self.list_images = [i for i in list_images]
        self.list_labels = [i.replace('imgs','masks').replace('crop','mask') for i in list_images]
        self.len = len(self.list_images)
        #'''
        
        # Get images path and labels
        # /mnt/DADOS_GRENOBLE_1/edemir/BRUMADINHO/1_PREPARE_DATA/results/1_PREPARE_DATA/raw_data/masks
        self.list_labels = indices.copy()
        self.list_images = [i.replace('masks', 'imgs').replace('mask_', 'crop_') for i in self.list_labels]
        self.len = len(self.list_images)
                        
        self.mean = None
        self.std = None
        self.mode = mode
        
        self.iscomputed = False
        
        # Create dict of classes
        self.num_classes = num_classes

        #self.weight_class = 1. / np.unique(np.array(self.list_labels), return_counts=True)[1]
        #self.samples_weights = self.weight_class[self.list_labels]
        
        #self.channels = dataset_info['channels']
        
        self.transform = []
        
        self.images = []
        self.labels = []
        
        self.memory_load = memory_load
        
        if self.memory_load:
            # Load all data in memory, however dont blame yourself after that.
            for img_path, label_path in zip(self.list_images, self.list_labels):
                #self.images.append(imread(img_path)[:, :, [0,1,2]])
                #self.images.append(imread(img_path)[:, :, :])
                #print(img_path)
                #print(label_path)
                
                if not isRGB:
                
                    #print("Using all bands RGBN in this dataloader")
                    img_tmp = imread(img_path)
                    dec_tmp = imread(img_path.replace('imgs','dec'))
                    mdt_tmp = imread(img_path.replace('imgs','mdt'))
                    
                    dec_tmp = np.expand_dims(dec_tmp, axis=2).copy()
                    mdt_tmp = np.expand_dims(mdt_tmp, axis=2).copy()
                    
                    self.images.append(np.concatenate((img_tmp, dec_tmp, mdt_tmp), axis=2))
                    
                    
                else:
                    #print("Using only NRG bands in this dataloader")
                    #print('Is RGB Mode')
                    # Geoeye bands expected order: 0123 <-> RGBN
                    # Use NRG instead RGB
                    img_tmp = imread(img_path)[:,:,[3,0,1]]
                    
                    # Last Chance, dont work well
                    #img_tmp = img_as_ubyte(exposure.rescale_intensity(img_tmp))
                    
                    #print(img_tmp)
                    self.images.append(img_tmp)
                    
                self.labels.append(imread(label_path)[:, :])
                
        
        print("Distribution")
        
        self.class_weights = []
        for i in range(self.num_classes):
            self.class_weights.append(0)
        
        # Compute Weight for class in segsemantic
        for index, mask in enumerate(self.labels):
        
            # Count each label pixel in image
            labels, count = np.unique(mask,return_counts=True)
            
            for l, c in zip(labels, count):
                if l != 254.0:
                    self.class_weights[int(l)] += c
                
                
        if True:
            # 1) Normalize distribution over proportion
            print('Normalize distribution over proportion')
            weight_sum = sum(self.class_weights)/self.class_weights
            weight_sum_norm = weight_sum/sum(weight_sum)
            self.class_weights = weight_sum_norm
            print('{} Class Weight: {}'.format(self.mode, self.class_weights))
        else:
            # 1) Normalize using Inverse of #samples
            print('Normalize Inverse of proportion sampler')
            self.class_weights = 1./np.array(self.class_weights)
            print('{} Class Weight: {}'.format(self.mode, self.class_weights))
        
    
        '''
        if mode == 'train':
        
            for img_path, x in zip(self.list_images, self.images):
            
                print(img_path)
                
                print('{} {}'.format(np.min(x[:,:,0]), np.max(x[:,:,0])))
                print('{} {}'.format(np.min(x[:,:,1]), np.max(x[:,:,1])))
                print('{} {}'.format(np.min(x[:,:,2]), np.max(x[:,:,2])))
                print('{} {}'.format(np.min(x[:,:,3]), np.max(x[:,:,3])))
                
                print('{} {}'.format(np.min(x[:,:,4]), np.max(x[:,:,4])))
                print('{} {}'.format(np.min(x[:,:,5]), np.max(x[:,:,5])))
                
            exit(1)
        '''
          
        
        
        self.random_augmentation = Remote_Seasing_Data_Augmentation(stack=False,
                                                                    rotate90=True,
                                                                    rotate60=False,
                                                                    rotate30=False,
                                                                    rotate180=True,
                                                                    fliph=True,
                                                                    flipv=True,
                                                                    zoom=False,
                                                                    shear=False,
                                                                    speckle_noise=False,
                                                                    #speckle_noise=True,
                                                                    translate=False) 
        
        
        
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        
        # Get new image/mask path
        img_path = self.list_images[index]
        lbl_path = self.list_labels[index]

        if not self.memory_load:
            # TODO: Maybe its time to load all dataset in memory to speedup
            # Load image and mask
            try:
                #Before was assign to np.uint conversion, but bad read on images
                img = imread(img_path)[:, :, :]#.astype(np.uint8)            
                mask = imread(lbl_path)[:, :]
                #OTHER DATA INPUTS
                print('NAO DEVE ENTRAR AQUI')
                
                
            except Exception as e:
                print('Error in load image: {}'.format(e))
                raise(e)
            
        else:
            img = self.images[index]
            mask = self.labels[index]
            
            #print(img.shape)
        
        #Just for binary problems. More than 2 classes need an original mask
        #mask[mask == 255] = 1
        
        '''
        # Random augmentation
        aug_choice = np.random.randint(5)
   
        if aug_choice == 0:
            # Rotate +90
            img = np.rot90(img).copy()
        elif aug_choice == 1:
            #Flip an array horizontally.
            img = np.fliplr(img).copy()
        elif aug_choice == 2:
            #Flip an array horizontally.
            img = np.flipud(img).copy()
        elif aug_choice == 3:
            # Rotate -90
            img = np.rot90(img,k=3).copy()
        '''
        
        # Already used for mean/std computation?
        if self.iscomputed:
            # Random crop
            #size=64
            size=128
            #if self.mode == 'train' or self.mode == 'val':
            if self.mode == 'train': #or self.mode == 'val':
                img, mask = random_crop(img,mask,size,size)
            
            # Do the augmentation over random crops
            if self.mode == 'train': 
                img, mask = self.random_augmentation.fit(img, mask)
                
        # Apply transform 
        if self.transform:
            #print("Using Transform")
            img = self.transform(img).float()
            mask = torch.from_numpy(mask).long()
                
        return img, mask, img_path
        
def online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = None
    snd_moment = None

    print("Computing mean of data...")
    for data_batch, _, data_batch_path in tqdm.tqdm(loader):
        b, c, h, w = data_batch.shape
        
        data_batch = data_batch.double()
           
        for it in range(data_batch.shape[0]):
               
            data = data_batch[it].view(1,c,h,w)
            
            # Dont use patches with nan value to compute mean/std
            if torch.min(data_batch[it][0,:,:]) == -9999.0:
                print('Amostras com NODATA')
            
            else:
            
                #print(data_batch_path[it])
                #print('{} {}'.format(torch.min(data_batch[it][0,:,:]), torch.max(data_batch[it][0,:,:])))
                #print('{} {}'.format(torch.min(data_batch[it][1,:,:]), torch.max(data_batch[it][1,:,:])))
                #print('{} {}'.format(torch.min(data_batch[it][2,:,:]), torch.max(data_batch[it][2,:,:])))
                #print('{} {}'.format(torch.min(data_batch[it][3,:,:]), torch.max(data_batch[it][3,:,:])))
                #print(data.shape)
                #print('{} {}'.format(torch.min(data_batch[it][4,:,:]), torch.max(data_batch[it][4,:,:])))
                #print('{} {}'.format(torch.min(data_batch[it][5,:,:]), torch.max(data_batch[it][5,:,:])))
                
            
                if fst_moment is None:
                    fst_moment = torch.empty(c, dtype=torch.double)
                    snd_moment = torch.empty(c, dtype=torch.double)

                nb_pixels = b * h * w

                sum_ = torch.sum(data, dim=[0, 2, 3])
                sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
                fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
                
                snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

                cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)

def get_transforms(loader, phase, mean=None, std=None):

    # Load dataloader for train and test
    loader.dataset.transform = transforms.Compose([#transforms.Resize(input_size), 
                                                    #transforms.CenterCrop(input_size), 
                                                    transforms.ToTensor()])
    
    if phase == 'train':
        
        if mean is None:
            mean, std = online_mean_and_sd(loader)
            print(mean,std)
        
        transform = transforms.Compose([#transforms.RandomResizedCrop(input_size),
                                                        #transforms.Resize(input_size), 
                                                        #transforms.RandomHorizontalFlip(), 
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean, std)
                                                        ])
    else:
        transform = transforms.Compose([#transforms.Resize(input_size), 
                                                        #transforms.CenterCrop(input_size), 
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean, std)
                                                        ])
    return mean, std, transform

class Remote_Seasing_Data_Augmentation():
    def __init__(self,
                 stack=True,
                 rotate90=True,
                 rotate60=True,
                 rotate30=True,
                 rotate180=True,
                 fliph=True,
                 flipv=True,
                 zoom=True,
                 shear=True,
                 speckle_noise=True,
                 translate=True):

        random.seed(42)

        self.stack = stack

        # Add all augmentation types
        self.augmentation = {0 : None}

        if rotate90:
            self.augmentation.update({1:self.__rotate90__})
        if rotate60:
            self.augmentation.update({2: self.__rotate60__})
        if rotate30:
            self.augmentation.update({3: self.__rotate30__})
        if rotate180:
            self.augmentation.update({4: self.__rotate180__})
        if fliph:
            self.augmentation.update({5: self.__fliph__})
        if flipv:
            self.augmentation.update({6: self.__flipv__})
        if zoom:
            self.augmentation.update({7: self.__zoom__})
        if shear:
            self.augmentation.update({8: self.__shear__})
        if speckle_noise:
            self.augmentation.update({9: self.__speckle_noise__})
        if translate:
            self.augmentation.update({10: self.__translate__})

        self.keys = list(self.augmentation.keys())

    def fit(self, img, mask):
        if self.stack:

            # Get the number of augmentations to use 
            num_augs = random.choice(range(1, 4))

            # Get which augmentarion to use not repeating 
            aug_choice = random.sample(self.keys, k=num_augs)

            for aug in aug_choice:
                if aug == 0:
                    pass
                else:
                    img = self.augmentation[aug](img, mask)

            return img 
            
        else:

            aug_choice = random.choice(self.keys)
            if aug_choice == 0:
                return img, mask 
            else:
                return self.augmentation[aug_choice](img, mask)


    def __rotate90__(self, img, mask):
        # Rotate +90 or -90
        if bool(random.getrandbits(1)):
            return np.rot90(img).copy(), np.rot90(mask).copy()
        else:
            return np.rot90(img,k=3).copy(), np.rot90(mask,k=3).copy()

    def __rotate60__(self, img, mask):
        # Rotate +60 or -60
        if bool(random.getrandbits(1)):
            return rotate(img, angle=60, reshape=False), rotate(mask, angle=60, reshape=False)
        else:
            return rotate(img, angle=-60, reshape=False), rotate(mask, angle=-60, reshape=False)

    def __rotate30__(self, img, mask):
        # Rotate +30 or -30
        if bool(random.getrandbits(1)):
            return rotate(img, angle=30, reshape=False), rotate(mask, angle=30, reshape=False)
        else:
            return rotate(img, angle=-30, reshape=False), rotate(mask, angle=-30, reshape=False)

    def __rotate180__(self, img, mask):
        # Rotate +180 or -180
        if bool(random.getrandbits(1)):
            return np.rot90(img, k=2).copy(), np.rot90(mask, k=2).copy()
        else:
            return np.rot90(img,k=4).copy(), np.rot90(mask,k=4).copy()

    def __fliph__(self, img, mask):
        #Flip an array horizontally.
        return np.fliplr(img).copy(), np.fliplr(mask).copy()

    def __flipv__(self, img, mask):
        #Flip an array horizontally.
        return np.flipud(img).copy(), np.flipud(mask).copy()

    def __zoom__(self, img, mask):
        #Zoom the array in 150% or 200%
        if bool(random.getrandbits(1)):
            return self.clipped_zoom(img, 1.5), self.clipped_zoom(mask, 1.5)
        else:
            return self.clipped_zoom(img, 2), self.clipped_zoom(mask, 1.5)
            
    def __shear__(self, img, mask):
        # Shearing the image from 0° to 18° or 0° to 36°
        if bool(random.getrandbits(1)):
            shear = iaa.Affine(shear=(0,18))
            return shear.augment_image(img), shear.augment_image(mask)
        else: 
            shear = iaa.Affine(shear=(0,36))
            return shear.augment_image(img), shear.augment_image(mask)

    def __speckle_noise__(self, img, mask):
        # Return a image with some speckle noise
        choice = np.random.randint(4)
        if choice == 0:
            return random_noise(img, mode='speckle', mean=0, var=0.001, clip=True), mask
        elif choice == 1:
            return random_noise(img, mode='speckle', mean=0, var=0.004, clip=True), mask
        elif choice == 2:   
            return random_noise(img, mode='speckle', mean=0, var=0.007, clip=True), mask
        else:
            return random_noise(img, mode='speckle', mean=0, var=0.010, clip=True), mask


    def __translate__(self, img, mask):
        # Translate the image 
        if bool(random.getrandbits(1)):
            translate = iaa.Affine(translate_percent={"x": (-0.25, 0.25), "y": (-0.25, 0.25)},
                                  mode=["reflect", "reflect"])
            return translate.augment_image(img), translate.augment_image(mask)
        else:
            translate = iaa.Affine(translate_percent={"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
                                  mode=["reflect", "reflect"])
            return translate.augment_image(img), translate.augment_image(mask)



    def clipped_zoom(self, img, zoom_factor, **kwargs):

        h, w = img.shape[:2]

        # For multichannel images we don't want to apply the zoom factor to the RGB
        # dimension, so instead we create a tuple of zoom factors, one per array
        # dimension, with 1's for any trailing dimensions after the width and height.
        zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

        # Zooming out
        if zoom_factor < 1:

            # Bounding box of the zoomed-out image within the output array
            zh = int(np.round(h * zoom_factor))
            zw = int(np.round(w * zoom_factor))
            top = (h - zh) // 2
            left = (w - zw) // 2

            # Zero-padding
            out = np.zeros_like(img)
            out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

        # Zooming in
        elif zoom_factor > 1:

            # Bounding box of the zoomed-in region within the input array
            zh = int(np.round(h / zoom_factor))
            zw = int(np.round(w / zoom_factor))
            top = (h - zh) // 2
            left = (w - zw) // 2

            out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

            # `out` might still be slightly larger than `img` due to rounding, so
            # trim off any extra pixels at the edges
            trim_top = ((out.shape[0] - h) // 2)
            trim_left = ((out.shape[1] - w) // 2)
            out = out[trim_top:trim_top+h, trim_left:trim_left+w]

        # If zoom_factor == 1, just return the input array
        else:
            out = img
        return out
