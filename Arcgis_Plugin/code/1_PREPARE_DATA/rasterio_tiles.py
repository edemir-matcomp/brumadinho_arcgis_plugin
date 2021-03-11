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

import argparse
import math, random, shutil
import numpy as np
from collections import Counter

from skimage import io
from skimage import transform as T
import rasterio
from rasterio.windows import Window
from skimage.io import imread, imsave

def get_classes(inRaster_Reference, bg_value=False):

    # Get number of classes
    mask = imread(inRaster_Reference)
    list_classes = list(np.unique(mask).astype(int))
    if bg_value:
        try:
            print('Removing value {}'.format(bg_value))
            list_classes.remove(int(bg_value))
        except:
            pass
        
    print('List of {} classes from Reference Raster: {}'.format(len(list_classes), list_classes))

    return sorted(list_classes)

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

# A function that returns the number of pixels from class:
def myFunc(e):
  return e[1]
    
def generate_tiles(inRaster_Image, inRaster_MDT, inRaster_DEC, inRaster_Reference, outRaster, size):

    if inRaster_MDT and inRaster_DEC:
        inRaster_MDT = rasterio.open(inRaster_MDT)
        inRaster_DEC = rasterio.open(inRaster_DEC)

    with rasterio.open(inRaster_Image) as src:

        # Get Info from original raster
        image = src.read()
        bands, cols,rows = image.shape
        kwargs = src.meta.copy()
        
        print('Raster shape: ',image.shape)
     
        # Read the input mask
        mask = io.imread(inRaster_Reference).astype('uint8')

        print('Mask Stats: ',Counter(mask.ravel()))

        total_w = int(math.ceil(cols/size))
        total_h = int(math.ceil(rows/size))

        print('Total dimensions: ', total_w, total_h)

        if not os.path.exists(outRaster):
            os.makedirs(outRaster)

        imgout = os.path.join(outRaster, 'imgs')
        maskout = os.path.join(outRaster, 'masks')

        mdtout = os.path.join(outRaster, 'mdt')
        decout = os.path.join(outRaster, 'dec')
        
        if not os.path.exists(imgout):
            os.mkdir(imgout)
        if not os.path.exists(maskout):
            os.mkdir(maskout)

        if inRaster_MDT:
            if not os.path.exists(mdtout):
                os.mkdir(mdtout)
        if inRaster_DEC:
            if not os.path.exists(decout):
                os.mkdir(decout)

        k = 0
        for i in range(total_w):
            for j in range(total_h):
            
                #crop_img = np.zeros((size, size, bands), dtype=img.dtype)
                #crop_mask = np.zeros((size, size), dtype=np.dtype('float32'))
                #crop_mask = np.zeros((size, size), dtype=np.dtype('uint8'))
                #crop_mask = np.zeros((size, size), dtype=image.dtype)
                crop_mask = np.full((size,size), 254, dtype=image.dtype)
                
                # rasterio window crop
                window_crop = Window.from_slices((i*size, min(i*size + size, cols)), (j*size, min(j*size + size, rows)))
                
                # Old way
                aux = mask[i*size:min(i*size + size, cols), j*size:min(j*size + size, rows)]
                crop_mask[:aux.shape[0], :aux.shape[1]] = aux[:,:]

                cnt = Counter(crop_mask.ravel())
           
                #if len(cnt) < 2 and cnt[254] != 0:
                #    continue
                #if float(cnt[255]) / (size*size) > 0.25:
                # Avoid patches with 75% of pixels with class 0 (background)
                #if float(cnt[254]) / (size*size) < 0.75:

                # AVOID 100% NaN
                # GET PATCHES ONLY WITH COUNT 254 EQUAL TO ZERO
                if float(cnt[254]) == 0:

                    # Save new image georeferenced tif
                    kwargs_image_tmp = src.meta.copy()
                    kwargs_image_tmp.update({
                        'height': window_crop.height,
                        'width': window_crop.width,
                        'transform': rasterio.windows.transform(window_crop, src.transform)})
                    
                    with rasterio.open(os.path.join(imgout, "crop_{}_{}_".format(i,j) + str(k) + ".tif"), 'w', **kwargs_image_tmp) as dst:
                        dst.write(src.read(window=window_crop))
                        
                    # Save new mask georeferenced tif
                    kwargs_mask_tmp = src.meta.copy()
                    kwargs_mask_tmp.update({
                        'count' : 1,
                        'height': window_crop.height,
                        'width': window_crop.width,
                        'transform': rasterio.windows.transform(window_crop, src.transform)})
                        
                    with rasterio.open(os.path.join(maskout, "mask_{}_{}_".format(i,j) + str(k) + ".tif"), 'w', **kwargs_mask_tmp) as dst_mask:
                        dst_mask.write(crop_mask, indexes=1)
                        #dst_mask.write(crop_mask, indexes=1)
                        
                    if inRaster_MDT:
                        # Clip MDT image
                        kwargs_mdt_tmp = inRaster_MDT.meta.copy()
                        kwargs_mdt_tmp.update({
                            'count' : 1,
                            'height': window_crop.height,
                            'width': window_crop.width,
                            'transform': rasterio.windows.transform(window_crop, inRaster_MDT.transform)})
                    
                        with rasterio.open(os.path.join(mdtout, "crop_{}_{}_".format(i,j) + str(k) + ".tif"), 'w', **kwargs_mdt_tmp) as dst_mdt:
                        #with rasterio.open(os.path.join(mdtout, "crop_{}_{}_".format(i,j) + str(k) + ".tif"), 'w', **MDT_Raster.meta.copy()) as dst_mdt:
                            dst_mdt.write(inRaster_MDT.read(window=window_crop))

                    if inRaster_DEC:
                        # Clip Declividade image
                        kwargs_dec_tmp = inRaster_DEC.meta.copy()
                        kwargs_dec_tmp.update({
                            'count' : 1,
                            'height': window_crop.height,
                            'width': window_crop.width,
                            'transform': rasterio.windows.transform(window_crop, inRaster_DEC.transform)})
                    
                        with rasterio.open(os.path.join(decout, "crop_{}_{}_".format(i,j) + str(k) + ".tif"), 'w', **kwargs_dec_tmp) as dst_dec:
                        #with rasterio.open(os.path.join(decout, "crop_{}_{}_".format(i,j) + str(k) + ".tif"), 'w', **DEC_Raster.meta.copy()) as dst_dec:
                            dst_dec.write(inRaster_DEC.read(window=window_crop))
                        
                    
                    k += 1
            
def create_folds(inRaster_Reference, base_output):

    # Read recently created image masks
    list_images = recursive_glob(os.path.join(base_output,'raw_data','masks'))

    # Get number of classes from raster Reference
    list_classes = get_classes(inRaster_Reference, 254)
    num_classes = len(list_classes) 

    dict_paths = {}
    dict_sum = []

    # Initialize empty lists
    for i in list_classes:
        dict_paths[i] = []
        dict_sum.append(0)

    for img_path in list_images:

        # Read Mask Image
        img = imread(img_path)

        # Count each label pixel in image
        labels, count = np.unique(img,return_counts=True)

        #print(labels)
        #print(count)
        # Fill dict with patches that contains at least one class
        for l, c in zip(labels, count):
            if l != 254.0:
                dict_paths[l].append((img_path, c))

    # Count total frequency of each class
    for key in dict_paths:
        for pair in dict_paths[key]:
            dict_sum[key] += pair[1]
        #print(key, dict_sum[key])

    sorted_classes = np.argsort(dict_sum)
    #print(np.argsort(dict_sum))
    #print(np.sort(dict_sum))

    K_FOLD = 5
    dict_fold = {}
    #summed = [0,0,0,0,0]
    summed = {}
    for id_fold in range(K_FOLD):
        dict_fold[id_fold] = []
    
    for i in list_classes:
        summed[i] = [0,0,0,0,0]

    # Usar set

    last_list = []
    #Iteracao partindo da classe de menor numero de amostras
    for label in sorted_classes:

        print("NUMBER OF SAMPLES FROM CLASS {}: {}".format(label, len(dict_paths[label])))
        # Para cada imagem que possui essa classe, distribuir em folds diferentes
        # A cada iteracao, verificar a quantidade de pixels da classe no fold
        # Sempre escolher fold que tenho a menor quantidade de amostras
        # Heuristica suficiente
    
        # Remove samples already used, look for path
        if len(last_list) != 0:
            for sample in last_list:
                for item in dict_paths[label]:
                    if item[0] == sample[0]:
                        try:
                            dict_paths[label].remove(item)
                        except:
                            continue
        
        # Sort by counter
        dict_paths[label].sort(reverse=True, key=myFunc)
        
        for index, res in enumerate(dict_paths[label]):
    
            #Check for every folds
            next_fold = np.argmin(summed[label])
      
            # Set on fold
            dict_fold[next_fold].append(res)
        
            summed[label][next_fold] += res[1]
        
        # Keep last list to remove in next iteration
        last_list += dict_paths[label].copy()
    
    #Save files
    fp = open(os.path.join(base_output,'stratified_folds'),'w+')
    for id_fold in range(K_FOLD):
        for img_path in dict_fold[id_fold]:
            fp.write('{},{}\n'.format(img_path[0],id_fold))

    
            
def main():
    parser = argparse.ArgumentParser(description='Create Raster Mask from Shapefile')
    parser.add_argument('--inRaster_Image', type=str, required=True,
                        help='Path to Raster Image')
    parser.add_argument('--inRaster_MDT', type=str, required=False, default='',
                        help='Path to MDT Image')
    parser.add_argument('--inRaster_DEC', type=str, required=False, default='',
                        help='Path to DEC Image')
    parser.add_argument('--inRaster_Reference', type=str, required=True,
                        help='Path to Raster Reference')
    parser.add_argument('--patchSize', type=int, default=256,
                        help='Size of Patches')
    parser.add_argument('--outRaster', type=str, required=True,
                        help='Path to folder where output Raster tiles will be saved')

    args = parser.parse_args()
    inRaster_Image = args.inRaster_Image
    #inRaster_MDT = args.inRaster_MDT
    #inRaster_DEC = args.inRaster_DEC
    inRaster_MDT = None if args.inRaster_MDT == 'None' else args.inRaster_MDT
    inRaster_DEC = None if args.inRaster_DEC == 'None' else args.inRaster_DEC
    inRaster_Reference = args.inRaster_Reference
    patchSize = args.patchSize
    outRaster = args.outRaster

    

    print(args)
    #Manage directories: Delete and Recreate Paths

    #Delete
    base_output = os.path.join(outRaster, '1_PREPARE_DATA')
    print(base_output)
    
    if os.path.exists(base_output):
         shutil.rmtree(base_output, ignore_errors=True)
            
    # Run process
    print('MDT',inRaster_MDT,type(inRaster_MDT))
    print('DEC:',inRaster_DEC,type(inRaster_DEC))
    raw_data = os.path.join(base_output,'raw_data')
    generate_tiles(inRaster_Image, inRaster_MDT, inRaster_DEC, inRaster_Reference, raw_data, patchSize)
    
    # Create folds using masks from generate tiles
    create_folds(inRaster_Reference, base_output)
    
if __name__ == '__main__':
    main()
    print('Preprocessing Data Complete!')
